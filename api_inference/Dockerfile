# Dockerfile for the inference API: builds Python image, installs deps, and launches FastAPI server
FROM python:3.10-slim

WORKDIR /app

# Copiar e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el servidor de inferencia y adaptadores
COPY inference_server.py .
COPY adapters ./adapters

# Exponer puerto de la API
EXPOSE 8000

# Arrancar servidor FastAPI
CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "8000"]
