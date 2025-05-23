# inference_server.py
# Purpose: Consume messages from Redis, classify with regex + PEFT model AND TF-IDF+Léxico,
#          save in MongoDB, broadcast via WebSocket, and accept moderator feedback.

import os
import re
import json
import asyncio
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from motor.motor_asyncio import AsyncIOMotorClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from fastapi.middleware.cors import CORSMiddleware
from bson import ObjectId

# --- Extra: Carga de modelo clásico TF-IDF+léxico ---
import joblib
import numpy as np
from scipy.sparse import hstack

# --- Load environment ---
load_dotenv()
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
MONGO_DB  = os.getenv("MONGO_DB", "moderation") 
MODEL_NAME = os.getenv("MODEL_NAME", "pysentimiento/robertuito-hate-speech")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./adapters/latest")
TFIDF_PATH = "models/tfidf_lexico"

# --- Initialize clients ---
redis = Redis.from_url(REDIS_URL)
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[MONGO_DB] 
collection = db.get_collection("messages")

# --- Load model & tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Attempt to load PEFT adapter if present
model = base_model
adapter_config = os.path.join(ADAPTER_PATH, "adapter_config.json")
if os.path.isdir(ADAPTER_PATH) and os.path.exists(adapter_config):
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        print(f"Loaded PEFT adapter from {ADAPTER_PATH}")
    except Exception as e:
        print(f"Failed to load PEFT adapter, continuing with base model: {e}")
else:
    print("No PEFT adapter found, using base model.")

# --- Load TF-IDF+Léxico model ---
try:
    vectorizer = joblib.load(os.path.join(TFIDF_PATH, "vectorizer.joblib"))
    clf_tfidf = joblib.load(os.path.join(TFIDF_PATH, "model.joblib"))
    lexicons = joblib.load(os.path.join(TFIDF_PATH, "lexicon.pkl"))
    print("TF-IDF+léxico model loaded correctly.")
except Exception as e:
    print(f"⚠️ Failed to load TF-IDF+léxico model: {e}")
    vectorizer = None
    clf_tfidf = None
    lexicons = None

def lexico_features(text, lexicons):
    words = set(text.lower().split())
    return np.array([len(words & lexicons[name]) for name in lexicons]).reshape(1, -1)

def predict_tfidf(text):
    if vectorizer is None or clf_tfidf is None or lexicons is None:
        return {"label": "unknown", "prob": 0.0}
    X_tfidf = vectorizer.transform([text])
    X_lexico = lexico_features(text, lexicons)
    X = hstack([X_tfidf, X_lexico])
    proba = clf_tfidf.predict_proba(X)[0]
    label_idx = np.argmax(proba)
    label = "toxic" if label_idx == 1 else "clean"
    score = float(proba[label_idx])
    return {"label": label, "prob": score}

# --- Compile regex patterns ---
regex_patterns = {
    "all_caps": re.compile(r'^[^a-z]*[A-ZÑÁÉÍÓÚ]{2,}[^a-z]*$'),
    "money_request": re.compile(r'\b(dona(me)?|regala(me)?|pago)\b', re.IGNORECASE),
    # TODO: add more patterns for spam, links, slurs, etc.
}

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes poner ["http://localhost:8080"] si solo usas ese origen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clients = set()

class Feedback(BaseModel):
    id: str = Field(..., description="MongoDB document _id")
    confirmado: bool = Field(..., description="True if message is banned")
    moderador: str = Field(None, description="Moderator username")

@app.on_event("startup")
async def startup_event():
    # Launch background inference worker
    asyncio.create_task(inference_worker())

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await ws.receive_text()  # mantiene viva la conexión
    except WebSocketDisconnect:
        clients.discard(ws)
        print("WebSocket disconnected, removed client")

async def broadcast(message: dict):
    living = set()
    for ws in clients:
        try:
            await ws.send_json(message)
            living.add(ws)
        except:
            pass
    clients.clear()
    clients.update(living)

@app.post("/feedback")
async def post_feedback(fb: Feedback):
    """
    Receive moderator feedback and update the corresponding message document.
    """
    try:
        oid = ObjectId(fb.id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    update = {
        "feedback": {
            "confirmado": fb.confirmado,
            "moderador": fb.moderador or "anonymous",
            "ts": datetime.utcnow().isoformat() + "Z"
        }
    }
    result = await collection.update_one({"_id": oid}, {"$set": update})

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"status": "ok", "updated": result.modified_count}

async def inference_worker():
    """
    Consume messages from Redis, clasifica con regex + modelo + tfidf_lexico,
    almacena en MongoDB y emite a clientes WebSocket.
    Nunca muere por errores de formato o excepciones puntuales.
    """
    while True:
        try:
            # 1) Espera bloqueante hasta un nuevo mensaje
            _, data = await redis.brpop("incoming")
            raw = data.decode()

            # 2) Parse seguro de JSON
            try:
                doc = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON: {e} – raw: {raw}")
                continue

            # 3) Clasificación: LoRA/regex + modelo propio
            label = None
            via = []
            prob = 0.0

            # 3a) Primero regex
            for name, pattern in regex_patterns.items():
                if pattern.search(doc["text"]):
                    label = name
                    via.append("regex")
                    prob = 1.0
                    break

            # 3b) Si no, el modelo LoRA/base
            if label is None:
                inputs = tokenizer(doc["text"], return_tensors="pt", truncation=True)
                outputs = model(**inputs)
                scores = outputs.logits.softmax(dim=-1).tolist()[0]
                toxic_prob = scores[1] if len(scores) > 1 else 0.0
                label = "toxic" if toxic_prob > 0.5 else "clean"
                via.append("model")
                prob = toxic_prob

            prediction_lora = {"label": label, "prob": prob, "via": via}
            # 3c) Modelo TF-IDF+léxico
            prediction_tfidf = predict_tfidf(doc["text"])

            # 4) Construye el documento Mongo
            document = {
                "channel":   doc.get("channel"),
                "user":      doc.get("user"),
                "text":      doc.get("text"),
                "timestamp": doc.get("timestamp"),
                "prediction_lora": prediction_lora,
                "prediction_tfidf": prediction_tfidf
            }

            # 5) Inserta y recupera el _id
            result = await collection.insert_one(document)
            document["_id"] = str(result.inserted_id)

            # 6) Emite a través de WebSocket
            await broadcast(document)

        except Exception as e:
            print(f"Error in inference_worker: {e}")
            await asyncio.sleep(1)

