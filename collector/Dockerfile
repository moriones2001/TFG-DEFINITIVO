# Dockerfile for the collector service: builds a Node.js image and runs collector.js
FROM node:18-alpine

WORKDIR /app

# Copiar y instalar dependencias de Node.js
COPY package.json package-lock.json* ./
RUN npm install --omit=dev


# Copiar el script del colector
COPY collector.js ./

# Comando por defecto para arrancar el servicio
CMD ["node", "collector.js"]
