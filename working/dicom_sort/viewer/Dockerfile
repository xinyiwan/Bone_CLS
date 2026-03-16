FROM python:3.10

ENV http_proxy="http://proxy.san.gva.es:8080"
ENV https_proxy="http://proxy.san.gva.es:8080"
ENV ftp_proxy="http://proxy.san.gva.es:8080"
ENV no_proxy="127.0.0.1,localhost"

# Tkinter + X11 client libraries needed to forward the GUI to the host display
# Xvfb is kept as a fallback for headless environments
RUN apt-get update && apt-get install -y \
    python3-tk \
    xvfb \
    libgl1-mesa-glx \
    libx11-6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /module

COPY . /module/

RUN pip install --no-cache-dir -r requirements.txt

# Start an interactive shell. After launching run one of:
#
#   Option A — X11 forwarding (host display visible):
#     docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ...
#     then inside: python PRMG_Visualizacion.py
#
#   Option B — virtual display with Xvfb (headless / no GUI needed):
#     inside container: xvfb-run -a python PRMG_Visualizacion.py
CMD ["bash"]
