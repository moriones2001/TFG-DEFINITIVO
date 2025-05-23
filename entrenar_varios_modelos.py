import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# ----- Preprocesado igual que antes -----
nltk.download('punkt')
nltk.download('stopwords')
spanish_stopwords = set(stopwords.words('spanish'))
stemmer = SnowballStemmer("spanish")

def emojis_to_words(text):
    text = emoji.demojize(text, delimiters=(" ", " "))
    replacements = {
        "joy": "risa", "heart": "corazon", "fire": "fuego",
        "angry": "enfado", "sad": "triste", "clap": "aplauso",
        "ok_hand": "ok", "thumbs_up": "positivo", "thumbs_down": "negativo",
        "laughing": "risa", "cry": "llora", "smile": "sonrisa",
        "scream": "susto", "poop": "caca"
    }
    for en, es in replacements.items():
        text = text.replace(f"{en}", es)
    text = text.replace(":", " ")
    return text

def clean_and_stem(text):
    text = text.lower().strip()
    text = emojis_to_words(text)
    tokens = nltk.word_tokenize(text, language="spanish")
    tokens = [stemmer.stem(w) for w in tokens if w not in spanish_stopwords and w.isalpha()]
    return " ".join(tokens)

# ----- Parámetros -----
DATASET_PATH = "data/dataset.csv"
LEXICON_NAMES = ['insultos', 'xenofobia', 'misoginia', 'inmigracion']
LEXICON_PATHS = [f"data/{name}.txt" for name in LEXICON_NAMES]
MODEL_DIR = "models/mejor_clasico"
RANDOM_STATE = 42
MIN_TEXT_LEN = 5
TEST_SIZE = 0.2

# ----- Carga y limpieza del dataset -----
print("Leyendo dataset...")
df = pd.read_csv(DATASET_PATH)
assert "text" in df.columns and "label" in df.columns, "CSV debe tener columnas 'text' y 'label'"

df = df.drop_duplicates(subset=["text"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > MIN_TEXT_LEN]
df = df[df["label"].isin([0, 1])]
df = df.dropna(subset=["text", "label"])
print(f"Ejemplos tras limpieza: {len(df)}")

# --- Normalización avanzada ---
print("Normalizando texto (emojis → palabras, stopwords, stemming)...")
df["text"] = df["text"].apply(clean_and_stem)

# --- Léxicos ---
LEXICONS = {}
for path, name in zip(LEXICON_PATHS, LEXICON_NAMES):
    with open(path, encoding="utf-8") as f:
        LEXICONS[name] = set(line.strip().lower() for line in f)
def lexico_features(text):
    words = set(text.split())
    return [len(words & LEXICONS[name]) for name in LEXICON_NAMES]

# --- Vectorización TF-IDF + léxico ---
vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1,3),
    stop_words=None
)
X_tfidf = vectorizer.fit_transform(df["text"])
X_lexico = np.array([lexico_features(txt) for txt in df["text"]])
X = hstack([X_tfidf, X_lexico])
y = df["label"].astype(int).values

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ----- Modelos a comparar -----
modelos = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE, solver="saga"),
    "RandomForest": RandomForestClassifier(n_estimators=120, class_weight='balanced', random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),  # no class_weight, pero suele ir bien
    "MultinomialNB": MultinomialNB(),
    "LinearSVC": LinearSVC(class_weight='balanced', max_iter=1500, random_state=RANDOM_STATE)
}

results = []
mejor_f1 = 0
mejor_modelo = None
mejor_nombre = None

print("\n=== COMPARATIVA DE MODELOS CLÁSICOS ===")
for nombre, clf in modelos.items():
    print(f"\n== Entrenando {nombre} ==")
    try:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        print(classification_report(y_test, y_pred, digits=3))
        results.append({
            "modelo": nombre,
            "f1": f1,
            "accuracy": acc
        })
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_modelo = clf
            mejor_nombre = nombre
    except Exception as e:
        print(f"Error entrenando {nombre}: {e}")

# --- Tabla resumen para la memoria ---
import pandas as pd
tabla = pd.DataFrame(results)
tabla = tabla.sort_values(by="f1", ascending=False)
print("\n=== Tabla comparativa de resultados ===")
print(tabla)
tabla.to_csv("modelos_clasicos_resultados.csv", index=False)

# --- Guarda el mejor modelo, vectorizador y léxicos ---
if mejor_f1 > 0.6:
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    joblib.dump(mejor_modelo, os.path.join(MODEL_DIR, "model.joblib"))
    joblib.dump(LEXICONS, os.path.join(MODEL_DIR, "lexicon.pkl"))
    print(f"\n✅ Mejor modelo: {mejor_nombre} (F1: {mejor_f1:.4f}) guardado en '{MODEL_DIR}'")
else:
    print("⚠️  Ningún modelo supera el umbral de F1=0.6. ¡Revisa tus datos!")

