FROM python:3.11-slim

ENV http_proxy="http://proxy.san.gva.es:8080"
ENV https_proxy="http://proxy.san.gva.es:8080"
ENV ftp_proxy="http://proxy.san.gva.es:8080"
ENV no_proxy="10.193.103.64,10.193.103.70"

ENV HTTP_PROXY="http://proxy.san.gva.es:8080"
ENV HTTPS_PROXY="http://proxy.san.gva.es:8080"
ENV FTP_PROXY="http://proxy.san.gva.es:8080"


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8579


HEALTHCHECK CMD curl --fail http://localhost:8579/_stcore/health || exit 1

# CMD ["streamlit", "run", "main.py", "--server.port=8517", "--server.address=0.0.0.0"]