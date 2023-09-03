FROM pytorch/torchserve:latest

WORKDIR /workspace

ENV BATCH_SIZE=32
ENV MAX_BATCH_DELAY=500

COPY chaser_ner_model.mar /workspace/chaser_ner_model.mar

EXPOSE 8080

CMD ["torchserve", "--start", "--model-store", "/workspace", "--models", "my_model=/workspace/chaser_ner_model.mar", "--batch_size", "$BATCH_SIZE", "--max_batch_delay", "$MAX_BATCH_DELAY"]
