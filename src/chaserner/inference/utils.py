import torch
from transformers import BertTokenizerFast
import json
from chaserner.model import NERModel
from chaserner.utils import model_output_to_label_tensor, extract_entities


def input_text_list_to_extracted_entities(input_text_list, config_path):
    with open(config_path) as f:
        config = json.load(f)
    ids2lbl = {v: k for k, v in config["lbl2ids"].items()}
    max_length = config["max_length"]
    model_path = config["best_checkpoint"]
    tokenizer_name = config["tokenizer_name"]
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    model = NERModel.load_from_checkpoint(checkpoint_path=model_path)

    tokenized_data = tokenizer(
        [txt.split() for txt in input_text_list],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
        is_split_into_words=True,
        return_offsets_mapping=True
    )

    outputs = model(tokenized_data["input_ids"], tokenized_data["attention_mask"])
    labels_list = model_output_to_label_tensor(outputs, tokenized_data["offset_mapping"], ids2lbl)
    return [{"input_text": input_text, "extracted_entities": {k: v for v, k in extract_entities(input_text.split(), labels)}} for input_text, labels in zip(input_text_list, labels_list)]


