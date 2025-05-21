# Dockerfile for the model_trainer service:
# installs dependencies and runs feedback extraction + adapter fine-tuning directly

FROM python:3.10-slim

WORKDIR /app

# Copiar e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar los scripts de extracción y entrenamiento
COPY extract_feedback.py .
COPY train_adapter.py .

# Comando por defecto: extraer feedback y luego entrenar adaptador
CMD ["sh", "-c", "python extract_feedback.py && python train_adapter.py"]
