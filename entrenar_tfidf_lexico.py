import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from scipy.sparse import hstack

import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Descargas solo si hace falta
nltk.download('punkt')
nltk.download('stopwords')

spanish_stopwords = set(stopwords.words('spanish'))
stemmer = SnowballStemmer("spanish")

# ---------------------------
# 1. PARÁMETROS
# ---------------------------
DATASET_PATH = "data/dataset.csv"
LEXICON_NAMES = ['insultos', 'xenofobia', 'misoginia', 'inmigracion']
LEXICON_PATHS = [f"data/{name}.txt" for name in LEXICON_NAMES]
MODEL_DIR = "models/tfidf_lexico"
RANDOM_STATE = 42
MIN_TEXT_LEN = 5
TEST_SIZE = 0.2

# ---------------------------
# FUNCIONES DE PREPROCESADO
# ---------------------------
def emojis_to_words(text):
    # Traduce emojis a alias tipo :joy:
    text = emoji.demojize(text, delimiters=(" ", " "))
    # Opcional: traduce alias principales a español (puedes añadir más)
    replacements = {
        "joy": "risa",
        "heart": "corazon",
        "fire": "fuego",
        "angry": "enfado",
        "sad": "triste",
        "clap": "aplauso",
        "ok_hand": "ok",
        "thumbs_up": "positivo",
        "thumbs_down": "negativo",
        "laughing": "risa",
        "cry": "llora",
        "smile": "sonrisa",
        "scream": "susto",
        "poop": "caca"
    }
    for en, es in replacements.items():
        text = text.replace(f"{en}", es)
    # Elimina los dos puntos de los alias emoji
    text = text.replace(":", " ")
    return text

def clean_and_stem(text):
    text = text.lower().strip()
    text = emojis_to_words(text)
    # Tokeniza y elimina stopwords, luego aplica stemming solo a las palabras (isalpha)
    tokens = nltk.word_tokenize(text, language="spanish")
    tokens = [stemmer.stem(w) for w in tokens if w not in spanish_stopwords and w.isalpha()]
    return " ".join(tokens)

# ---------------------------
# 2. CARGA Y LIMPIEZA DEL DATASET
# ---------------------------
print("Leyendo dataset...")
df = pd.read_csv(DATASET_PATH)
assert "text" in df.columns and "label" in df.columns, "CSV debe tener columnas 'text' y 'label'"

df = df.drop_duplicates(subset=["text"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > MIN_TEXT_LEN]
df = df[df["label"].isin([0, 1])]
df = df.dropna(subset=["text", "label"])
print(f"Ejemplos tras limpieza: {len(df)}")

# ---------------------------
# 2b. NORMALIZACIÓN AVANZADA DE TEXTO
# ---------------------------
print("Normalizando texto (emojis → palabras, stopwords, stemming)...")
df["text"] = df["text"].apply(clean_and_stem)

# Muestra ejemplos de texto normalizado
print("\n=== Ejemplos de texto normalizado ===")
for i in range(5):
    print(f"Original: {df.iloc[i]['text']}")
# ---------------------------
# 3. CARGA DE LÉXICOS
# ---------------------------
print("Cargando léxicos...")
LEXICONS = {}
for path, name in zip(LEXICON_PATHS, LEXICON_NAMES):
    with open(path, encoding="utf-8") as f:
        LEXICONS[name] = set(line.strip().lower() for line in f)

def lexico_features(text):
    words = set(text.split())
    return [len(words & LEXICONS[name]) for name in LEXICON_NAMES]

# ---------------------------
# 4. VECTORIZACIÓN TF-IDF + LÉXICO
# ---------------------------
print("Vectorizando texto...")
vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1,3),
    stop_words=None  # Ya quitamos stopwords en el preprocesado
)

X_tfidf = vectorizer.fit_transform(df["text"])
X_lexico = np.array([lexico_features(txt) for txt in df["text"]])
X = hstack([X_tfidf, X_lexico])
y = df["label"].astype(int).values

# ---------------------------
# 5. TRAIN/TEST SPLIT
# ---------------------------
print("Dividiendo en train y test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------
# 6. BÚSQUEDA DE HIPERPARÁMETROS
# ---------------------------
print("Buscando mejores hiperparámetros...")
param_grid = {
    'C': [0.1, 1, 3, 10],
    'solver': ['liblinear', 'saga']
}
grid = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
    param_grid,
    scoring='f1',
    cv=4,
    n_jobs=-1
)
grid.fit(X_train, y_train)
print(f"Mejor combinación: {grid.best_params_}")

# ---------------------------
# 7. EVALUACIÓN DEL MODELO
# ---------------------------
model = grid.best_estimator_

print("Evaluando en test...")
y_pred = model.predict(X_test)
print("\nMATRIZ DE CONFUSIÓN:\n", confusion_matrix(y_test, y_pred))
print("\nCLASSIFICATION REPORT:\n", classification_report(y_test, y_pred, digits=4))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

# ---------------------------
# 8. GUARDADO DEL MODELO SI ES BUENO
# ---------------------------
if f1_score(y_test, y_pred) > 0.6:  # puedes ajustar el umbral
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))
    joblib.dump(LEXICONS, os.path.join(MODEL_DIR, "lexicon.pkl"))
    print(f"\n✅ Modelo, vectorizador y léxicos guardados en '{MODEL_DIR}'")
else:
    print("⚠️  Modelo no se guarda porque el F1 es bajo. ¡Revisa los datos o prueba otros hiperparámetros!")

# ---------------------------
# 9. (Opcional) SAVE VALIDATION RESULTS
# ---------------------------
report_path = os.path.join(MODEL_DIR, "eval.txt")
with open(report_path, "w") as f:
    f.write("=== VALIDATION REPORT ===\n\n")
    f.write(f"Matriz de confusión:\n{confusion_matrix(y_test, y_pred)}\n\n")
    f.write(f"Classification report:\n{classification_report(y_test, y_pred, digits=4)}\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    f.write(f"F1-score: {f1_score(y_test, y_pred)}\n")
print(f"Informe de validación guardado en {report_path}")
