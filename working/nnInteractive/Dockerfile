# Use python:3.11-slim for a much smaller base image
FROM python:3.11-slim

ENV http_proxy="http://proxy.san.gva.es:8080"
ENV https_proxy="http://proxy.san.gva.es:8080"
ENV ftp_proxy="http://proxy.san.gva.es:8080"
ENV no_proxy="127.0.0.1,localhost"

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /working

# Install system dependencies, PyTorch, and application packages in one layer
# This minimizes intermediate layers and reduces final image size
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        xvfb \
        ca-certificates \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "numpy<2.0.0" \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch \
    && pip install --no-cache-dir fury \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN echo "force rebuild from here (2)"

# Copy application and install
COPY . /working


RUN pip install --no-cache-dir -r requirements.txt
RUN pip install nninteractive-slicer-server

# expose not needed if using -p
# If using only expose and not -p then will not work
# EXPOSE 80

CMD ["nninteractive-slicer-server", "--host 0.0.0.0", "--port 1527"]