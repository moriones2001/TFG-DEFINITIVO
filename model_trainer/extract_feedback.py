#!/usr/bin/env python3
# extract_feedback.py
# Purpose: Pull validated messages from MongoDB and write them to dataset.jsonl

import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient

# Carga variables de entorno (.env en la raíz con MONGO_URI, etc.)
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
DB_NAME = os.getenv("MONGO_DB", "moderation")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "messages")

def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    coll = db[COLLECTION_NAME]

    # Consulta todos los documentos que tengan campo 'feedback.confirmado'
    cursor = coll.find({"feedback.confirmado": {"$exists": True}},
                       {"text": 1, "feedback.confirmado": 1, "_id": 0})

    count = 0
    with open("dataset.jsonl", "w", encoding="utf-8") as f:
        for doc in cursor:
            label = 1 if doc["feedback"]["confirmado"] else 0
            entry = {"text": doc["text"], "label": label}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"Extracted {count} examples to dataset.jsonl")

if __name__ == "__main__":
    main()
