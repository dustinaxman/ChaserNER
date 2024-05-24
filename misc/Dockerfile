# FROM pytorch/torchserve:latest
# FROM pandaczm/torchserve_m1
FROM pandaczm/torchserve_m1:transformers_4.10.0

WORKDIR /workspace

ENV BATCH_SIZE=32
ENV MAX_BATCH_DELAY=500
#ENV PYTHONPATH="/workspace/ChaserNER/src/:${PYTHONPATH}"

#ENV DEBIAN_FRONTEND=noninteractive

USER root
#RUN apt-get update && \
#    apt-get install -y jq && \
#    rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size


COPY chaser_ner_model.mar /workspace/chaser_ner_model.mar
COPY config.properties /workspace/config.properties
COPY chaser_ner_model_config.yaml /workspace/chaser_ner_model_config.yaml
#COPY . /workspace/
EXPOSE 8080

#RUN chmod -R 777 /workspace/
#RUN pip install torchvision==0.18.0
#RUN pip install torch==2.3.0
#RUN pip install numpy==1.26.4
#RUN pip install pytorch-lightning==2.0.7
#RUN pip install datasets==2.14.4
#RUN pip install torchserve==0.6.0
#RUN pip install torchmetrics=1.1.0

RUN pip install transformers
#USER root
#RUN apt-get update && apt-get install -y git
#RUN cd /workspace && git clone https://github.com/huggingface/transformers.git

#RUN cd /workspace/transformers && git fetch origin pull/27734/head:fix-export-deberta-to-torchscript
#RUN cd /workspace/transformers && git checkout fix-export-deberta-to-torchscript
#RUN cd /workspace/transformers && pip install -e .

RUN pip install pytorch_lightning
RUN pip install datasets
RUN pip install seqeval

#RUN ls -la /workspace/ChaserNER/src/
#RUN ls -la /workspace/ChaserNER/src/chaserner/

#RUN python3 /workspace/ChaserNER/bin/insert_torchscript.py --config_path /workspace/config.json
#RUN /workspace/ChaserNER/bin/insert_torchserve.sh /workspace/

CMD ["torchserve", "--start", "--model-store", "/workspace", "--ts-config", "/workspace/config.properties", "--models", "chaser_ner_model=/workspace/chaser_ner_model.mar"]