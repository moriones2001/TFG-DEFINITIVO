# descargar_csv.py
from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

print("⬇️ Descargando dataset de Manueltonneau...")
ds = load_dataset("manueltonneau/spanish-hate-speech-superset", split="train")
df = ds.to_pandas()[["text", "labels"]].rename(columns={"labels": "label"})

df = df.dropna(subset=["text", "label"])
df = df[df["label"].isin([0, 1])]
df["text"] = df["text"].astype(str)

out_path = "data/dataset.csv"
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"✅ Dataset guardado en '{out_path}' con {len(df)} ejemplos.")
