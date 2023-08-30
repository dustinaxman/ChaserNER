import json
from typing import List, Dict
import torch

def join_raw_labels(raw_labels, offset_mapping):
    mask = (offset_mapping[:, :, 0] == 0) & (offset_mapping[:, :, 1] != 0)
    selected_values_list = [raw_labels[i][mask[i]] for i in range(mask.size(0))]
    return selected_values_list


def model_output_to_label_tensor(outputs, offset_mapping, ids2lbl):
    raw_labels = torch.argmax(outputs.logits, dim=-1)
    selected_values_list = join_raw_labels(raw_labels, offset_mapping)
    labels_list = [[ids2lbl[idx.item()] for idx in tensor] for tensor in selected_values_list]
    return labels_list


def batch_to_jsonl(batch, tokenizer, ids2lbl, outputs=None) -> List[str]:
    input_ids = batch['input_ids']
    raw_labels = batch['labels']
    offset_mapping = batch['offset_mapping'].squeeze(1)

    if outputs is not None:
        logits = outputs.logits
        all_predicted_classes = torch.argmax(logits, dim=-1)

    # Convert input_ids to tokens
    tok_texts = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

    # Convert tokens to raw_text (by joining tokens and removing special tokens)
    raw_texts = [' '.join([tok for tok in tok_text if not tok.startswith('[') and not tok.endswith(']')]) for tok_text
                 in tok_texts]

    # Convert numerical labels and predictions to their string representations
    gt_tok_forms = [[ids2lbl[idx.item()] for idx in tensor] for tensor in raw_labels]
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

    # Create the JSONL formatted data
    jsonl_data = []
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
        jsonl_data.append(json.dumps(entry))

    return jsonl_data

