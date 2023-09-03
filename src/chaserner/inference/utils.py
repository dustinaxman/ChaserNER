import torch
from transformers import BertTokenizerFast
import json
from chaserner.model import NERModel
from chaserner.utils import model_output_to_label_tensor, extract_entities
from pathlib import Path
import time

#import os
#os.environ["OMP_NUM_THREADS"] = "1"


def load_model(config_path, device):
    config_path = Path(config_path)
    with open(config_path) as f:
        config = json.load(f)
    ids2lbl = {v: k for k, v in config["lbl2ids"].items()}
    max_length = config["max_length"]
    model_path = config_path.parent / config["best_checkpoint"]
    tokenizer_name = config["tokenizer_name"]
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    # TODO: remove the extra args here later!!! for later models
    if "torchscript_model" in config:
        model_path = config_path.parent / config["torchscript_model"]
        print("LOADING TORCHSCRIPT FILE")
        model = torch.jit.load(str(model_path))
    else:
        print("LOADING MODEL CHECKPOINT")
        model = NERModel.load_from_checkpoint(checkpoint_path=model_path, hf_model_name=tokenizer_name,
                                              label_to_id=config["lbl2ids"])
    model.eval()
    model = model.to(device)
    return model, tokenizer, max_length, ids2lbl


def run_ner_model(input_text_list, model, tokenizer, max_length, ids2lbl, device):
    token_lengths = [len(tokenizer.tokenize(txt)) for txt in input_text_list]

    # Find the maximum token length from the tokenized texts
    max_input_length = max(token_lengths)

    max_length = min(max_input_length, max_length)

    tokenized_data = tokenizer(
        [txt.split() for txt in input_text_list],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
        is_split_into_words=True,
        return_offsets_mapping=True
    ).to(device)
    start_time_model_only = time.time()
    outputs = model(tokenized_data["input_ids"], tokenized_data["attention_mask"])
    total_time = time.time() - start_time_model_only
    print(f"MODEL ONLY: {total_time}")
    # print(outputs)
    labels_list = model_output_to_label_tensor(outputs, tokenized_data["offset_mapping"], ids2lbl)
    entity_extracted_samples = [{"input_text": input_text,
                                 "extracted_entities": {k: v for v, k in extract_entities(input_text.split(), labels)}}
                                for input_text, labels in zip(input_text_list, labels_list)]
    return entity_extracted_samples


def input_text_list_to_extracted_entities(input_text_list, config_path, device):
    start_time_1 = time.time()
    config_path = Path(config_path)
    model, tokenizer, max_length, ids2lbl = load_model(config_path, device)
    print(f"Using device: {device}")

    #max_input_length = max([len(txt.split()) for txt in input_text_list])

    start_time_2 = time.time()

    entity_extracted_samples = run_ner_model(input_text_list, model, tokenizer, max_length, ids2lbl, device)

    total_time = time.time() - start_time_2
    load_time = start_time_2 - start_time_1
    per_utt_time = total_time/float(len(input_text_list))
    print(f"Total time: {total_time}\nPer utt time: {per_utt_time}\nTotal utts: {len(input_text_list)}\nLoad time: {load_time}")
    # for input_text, labels in zip(input_text_list, labels_list):
    #     print(input_text)
    #     print(labels)
    return entity_extracted_samples


