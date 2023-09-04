# FROM pytorch/torchserve:latest
# FROM pandaczm/torchserve_m1
FROM pandaczm/torchserve_m1:transformers_4.10.0

WORKDIR /workspace

ENV BATCH_SIZE=32
ENV MAX_BATCH_DELAY=500

COPY chaser_ner_model.mar /workspace/chaser_ner_model.mar
COPY config.properties /workspace/config.properties
COPY chaser_ner_model_config.yaml /workspace/chaser_ner_model_config.yaml
EXPOSE 8080

RUN pip install transformers
RUN pip install pytorch_lightning
RUN pip install datasets
RUN pip install seqeval

CMD ["torchserve", "--start", "--model-store", "/workspace", "--ts-config", "/workspace/config.properties", "--models", "chaser_ner_model=/workspace/chaser_ner_model.mar"]