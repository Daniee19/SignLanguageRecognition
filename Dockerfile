# Imagen base con Python preinstalado
FROM python:3.10

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia todos los archivos del proyecto al contenedor
COPY . .

# Instala las dependencias (las agregaremos en el paso 3)
RUN pip install --no-cache-dir -r requirements.txt

# Ejecuta el script principal de tu proyecto
CMD ["python", "Sign_language_detection.py"]
