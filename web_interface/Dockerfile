# Dockerfile for the static web interface: uses Nginx to serve HTML/CSS/JS
FROM nginx:alpine

WORKDIR /usr/share/nginx/html

# Elimina la configuración por defecto
RUN rm -rf ./*

# Copia los archivos de la UI
COPY index.html .
COPY styles.css .
COPY script.js .
COPY default-avatar.png .   
# Exponemos el puerto 80 (mapeado en docker-compose a 8080)
EXPOSE 80
