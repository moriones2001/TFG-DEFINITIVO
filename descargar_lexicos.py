from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)

# Mapeo real del dataset
LEXICONS = {
    "insultos": "insults",
    "xenofobia": "xenophobia",
    "misoginia": "misogyny",
    "inmigracion": "inmigrant"
}

print("⬇️ Descargando léxicos de SINAI...")
ds = load_dataset("SINAI/hate-speech-spanish-lexicons")

for name, split in LEXICONS.items():
    words = ds[split]["text"]
    path = f"data/{name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(words))
    print(f"✅ Guardado: {path} ({len(words)} palabras)")

print("🎉 Léxicos descargados correctamente.")
