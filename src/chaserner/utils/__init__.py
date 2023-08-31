from typing import List, Dict
import torch
from transformers import BertTokenizerFast
import json
from chaserner.model import NERModel


def extract_entities(tokens, labels):
    entities = []
    entity = []
    entity_type = None

    for token, label in zip(tokens, labels):
    #id_to_label = {i: label for label, i in label_to_id.items()}
        if label.startswith("B-"):
            if entity:
                entities.append((' '.join(entity), entity_type))
                entity = []
            entity_type = label.split("-")[1]
            entity.append(token)
        elif label.startswith("I-"):
            current_entity_type = label.split("-")[1]
            if current_entity_type == entity_type:
                entity.append(token)
            else:
                if entity:
                    entities.append((' '.join(entity), entity_type))
                    entity = []
                entity_type = current_entity_type
                entity.append(token)
        else:
            if entity:
                entities.append((' '.join(entity), entity_type))
                entity = []

    # Catch any remaining entities
    if entity:
        entities.append((' '.join(entity), entity_type))

    return entities



def join_raw_labels(raw_labels, offset_mapping):
    mask = (offset_mapping[:, :, 0] == 0) & (offset_mapping[:, :, 1] != 0)
    selected_values_list = [raw_labels[i][mask[i]] for i in range(mask.size(0))]
    return selected_values_list


def model_output_to_label_tensor(outputs, offset_mapping, ids2lbl):
    raw_labels = torch.argmax(outputs.logits, dim=-1)
    selected_values_list = join_raw_labels(raw_labels, offset_mapping)
    labels_list = [[ids2lbl[idx.item()] for idx in tensor] for tensor in selected_values_list]
    return labels_list


def detokenize(tokenized_text):
    # Remove [CLS] and [SEP] tokens
    tokenized_text = [t for t in tokenized_text.split() if t not in ['[CLS]', '[SEP]']]
    # Convert WordPiece tokenization to full words
    text = ' '.join(tokenized_text).replace(' ##', '')
    # Handle punctuation and certain characters
    text = text.replace(" ' s", "'s")  # Handle possessives
    for punctuation in [' ,', ' .', ' ?', ' !', ' ;', ' :']:
        text = text.replace(punctuation, punctuation[-1])
    text = text.replace(" ' re", "'re") # Handle contractions
    text = text.replace(" / ", "/")  # Remove spaces around slashes
    return text


def batch_to_info(batch, tokenizer, ids2lbl, outputs=None) -> List[str]:
    input_ids = batch['input_ids']
    raw_labels = batch['labels']
    offset_mapping = batch['offset_mapping'].squeeze(1)

    if outputs is not None:
        logits = outputs.logits
        all_predicted_classes = torch.argmax(logits, dim=-1)

    # Convert input_ids to tokens
    tok_texts = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

    # Convert tokens to raw_text (by joining tokens and removing special tokens)
    raw_texts = [detokenize(" ".join(tok_text)) for tok_text in tok_texts]

    # Convert numerical labels and predictions to their string representations
    gt_tok_forms = [[ids2lbl[idx.item()] if idx != -100 else "-100" for idx in tensor] for tensor in raw_labels]
    if outputs is not None:
        hyp_tok_forms = [[ids2lbl[idx.item()] for idx in tensor] for tensor in all_predicted_classes]
    else:
        hyp_tok_forms = [["None"]] * len(gt_tok_forms)

    # Get normalized forms
    gt_norm_forms = join_raw_labels(raw_labels, offset_mapping)
    gt_norm_forms = [[ids2lbl[idx.item()] for idx in tensor] for tensor in gt_norm_forms]

    if outputs is not None:
        hyp_norm_forms = join_raw_labels(all_predicted_classes, offset_mapping)
        hyp_norm_forms = [[ids2lbl[idx.item()] for idx in tensor] for tensor in hyp_norm_forms]
    else:
        hyp_norm_forms = [["None"]] * len(gt_norm_forms)

    logged_info = []
    for raw_text, tok_text, gt_tok_form, gt_norm_form, hyp_tok_form, hyp_norm_form in zip(raw_texts, tok_texts,
                                                                                          gt_tok_forms, gt_norm_forms,
                                                                                          hyp_tok_forms,
                                                                                          hyp_norm_forms):
        entry = {
            "raw_text": raw_text,
            "tok_text": ' '.join(tok_text),
            "gt_tok_form": ' '.join(gt_tok_form),
            "gt_norm_form": ' '.join(gt_norm_form),
            "hyp_tok_form": ' '.join(hyp_tok_form),
            "hyp_norm_form": ' '.join(hyp_norm_form)
        }
        logged_info.append(entry)

    return logged_info


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

