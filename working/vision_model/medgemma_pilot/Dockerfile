# MedGemma 1.5 served locally via vLLM (OpenAI-compatible API on :9586).
#
#   Build:  docker build -t medgemma-vllm --build-arg HF_TOKEN=hf_xxx .
#   Run:    docker run --rm -it --gpus 6 -p 9586:9586 medgemma-vllm:latest
#
# Requires on the host: NVIDIA driver 550.x + nvidia-container-toolkit.
# Then point the script at it:
#   python run_medgemma.py --mode infer --backend openai \
#       --base-url http://localhost:9586/v1 --model-id google/medgemma-1.5-4b-it ...

FROM python:3.11-slim

# --- corporate proxy ---
ENV http_proxy="http://proxy.san.gva.es:8080"
ENV https_proxy="http://proxy.san.gva.es:8080"
ENV ftp_proxy="http://proxy.san.gva.es:8080"
ENV no_proxy="127.0.0.1,localhost"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl wget vim libnuma1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Pin vLLM to a version whose default wheel targets CUDA 12.4 (torch==2.6.0+cu124),
# which matches driver 550.144.03. Do NOT bump this without re-checking the
# CUDA build it pulls -- later vLLM lines move to newer default CUDA toolkits
# (12.6/12.8+) that require driver >=560 and will fail on this machine.
RUN pip install --no-cache-dir "vllm==0.8.5.post1" \
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    sentencepiece==0.2.0 \
    protobuf==5.28.3 \
    "huggingface_hub>=0.26.2"

# Client/eval deps (so you can also run run_medgemma.py --backend openai in-container).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH="/app:${PYTHONPATH}"

# MedGemma is a GATED model under the Health AI Developer Foundations (HAI-DEF)
# license: the token must have accepted access at
#   https://huggingface.co/google/medgemma-1.5-4b-it
# Pass it at build time; DO NOT commit a real token:
#   docker build --build-arg HF_TOKEN=hf_xxx ...
ARG HF_TOKEN
ENV MODEL_DIR=/models/medgemma-1.5-4b-it
# Download into a fixed local dir so we can patch the config before serving.
# NB: `huggingface-cli login` is deprecated and exits non-zero in current
# huggingface_hub -- use `hf download --token` directly (no separate login step).
RUN hf download google/medgemma-1.5-4b-it --token "$HF_TOKEN" --local-dir "$MODEL_DIR"

# vLLM 0.8.5's rope patcher (patch_rope_scaling_dict) requires every rope_scaling
# dict to carry a "rope_type" key. MedGemma 1.5's text_config.rope_scaling ships
# without one, so vLLM raises before the model loads:
#   ValueError: rope_scaling should have a 'rope_type' key
# Inject the missing key on disk. Gemma-3 uses "linear" scaling -> keeps the
# model's extended context. If you only send short prompts and would rather not
# assume a scaling type, change 'linear' below to None to disable scaling instead.
RUN python -c "import json,os; p=os.path.join(os.environ['MODEL_DIR'],'config.json'); c=json.load(open(p)); tc=c.get('text_config',c); rs=tc.get('rope_scaling'); ok=bool(rs) and 'rope_type' not in rs; rs and rs.setdefault('rope_type','linear'); ok and json.dump(c,open(p,'w'),indent=2); print('patched' if ok else 'no patch needed',':',rs)"

RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

EXPOSE 9586

# Serve from the patched local dir. --served-model-name keeps the OpenAI-facing
# id as "google/medgemma-1.5-4b-it" so run_medgemma.py --model-id is unchanged.
# sh -c so $MODEL_DIR expands (JSON exec-form does no env substitution).
#  - bf16 as requested (80GB GPU -> no quantization).
#  - Multimodal: vLLM 0.8.5 auto-detects Gemma-3 vision. If images get rejected,
#    add:  --limit-mm-per-prompt image=1
CMD ["sh", "-c", "vllm serve \"$MODEL_DIR\" \
     --served-model-name google/medgemma-1.5-4b-it \
     --host 0.0.0.0 --port 9586 --dtype bfloat16"]
