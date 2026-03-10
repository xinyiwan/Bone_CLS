# IMAGEN BASE
FROM tensorflow/tensorflow:2.10.0-gpu

# Se ponen los proxys para que funcione en IISLAFE
ENV http_proxy "http://proxy.san.gva.es:8080"
ENV https_proxy "http://proxy.san.gva.es:8080"
ENV no_proxy "127.0.0.1,localhost"


WORKDIR /module


RUN apt-get -y update
RUN apt-get install -y git
RUN apt-get install -y libgl1
RUN pip install --upgrade pip

COPY . /module/

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python", "Classifier_main.py"]

