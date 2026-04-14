# Use python:3.11-slim for a much smaller base image
FROM python:3.11-slim

ENV http_proxy="http://proxy.san.gva.es:8080"
ENV https_proxy="http://proxy.san.gva.es:8080"
ENV ftp_proxy="http://proxy.san.gva.es:8080"
ENV no_proxy="127.0.0.1,localhost"

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

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
    && pip install --no-cache-dir fury \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy application and install TotalSegmentator (lets it resolve its own deps)
COPY . /app
RUN pip install --no-cache-dir /app \
    && python /app/totalsegmentator/download_pretrained_weights.py \
    && find /usr/local/lib/python3.11 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.11 -type f -name "*.pyc" -delete \
    && find /usr/local/lib/python3.11 -type f -name "*.pyo" -delete

RUN pip install totalsegmentator

# Install PyTorch with GPU support ONCE, after all other deps are resolved.
# Use cu121 if your host driver supports up to CUDA 12.1 (nvidia-smi shows "CUDA Version: 12.1")
# Use cu124 if your host driver supports CUDA 12.4+ (driver >= 550.x)
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu124
# expose not needed if using -p
# If using only expose and not -p then will not work
# EXPOSE 80

CMD ["python", "run_totalseg.py", "/data/", "--out_dir", "/output/"]