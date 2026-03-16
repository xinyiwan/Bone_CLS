FROM python:3.10

ENV http_proxy="http://proxy.san.gva.es:8080"
ENV https_proxy="http://proxy.san.gva.es:8080"
ENV ftp_proxy="http://proxy.san.gva.es:8080"
ENV no_proxy="127.0.0.1,localhost"

# Instalar Tkinter y Xvfb
RUN apt-get update && apt-get install -y \
    python3-tk \
    xvfb \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Definir directorio de trabajo
WORKDIR /module

# Copiar archivos
COPY . /module/

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer puerto si es necesario
EXPOSE 8502

# Usar Xvfb para ejecutar la aplicación sin pantalla física
# CMD ["xvfb-run", "-a", "python", "main_opt.py"]