from typing import List, Dict
import torch
import copy
from collections import defaultdict
import torch.nn.functional as F

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.key_freq = {}
        self.freq_list = {}
        self.min_freq = 0
    def _update(self, key: int):
        freq = self.key_freq[key]
        self.key_freq[key] = freq + 1
        # remove the key from the current frequency list
        self.freq_list[freq].remove(key)
        if not self.freq_list[freq]:
            del self.freq_list[freq]
            # update the minimum frequency if needed
            if self.min_freq == freq:
                self.min_freq += 1
        # add the key to the new frequency list
        self.freq_list.setdefault(freq + 1, []).append(key)
    def get(self, key: int) -> int:
        # update the key's frequency
        if key not in self.cache:
            return None
        self._update(key)
        return self.cache[key]
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache[key] = value
            self._update(key)
            return
        # if the cache is full, evict the least frequently used key
        if len(self.cache) == self.capacity:
            lfu_key = self.freq_list[self.min_freq][0]
            del self.cache[lfu_key]
            del self.key_freq[lfu_key]
            self.freq_list[self.min_freq].pop(0)
            if not self.freq_list[self.min_freq]:
                del self.freq_list[self.min_freq]
        self.cache[key] = value
        self.key_freq[key] = 1
        self.freq_list.setdefault(1, []).append(key)
        self.min_freq = 1

def process_entities_to_dict(entities):
    entity_dict = defaultdict(list)
    for v, k in entities:
        entity_dict[k].append(v)
    return entity_dict

def strip_date_person_from_right(entities):
    # THEN
    # return the set of entity blocks up until that point
    head_entities = []
    tail_entities = []
    tail_flag = False
    for entity in entities:
        if not tail_flag:
            head_entities.append(entity)
            if entity[1] == "task":
                tail_flag = True
        else:
            tail_entities.append(entity)
    num_person_blocks = len([entity for entity in tail_entities if entity[1] == "person"])
    num_date_blocks = len([entity for entity in tail_entities if entity[1] == "date"])
    num_other_blocks = len([entity for entity in tail_entities if entity[1] == "O"])
    len_person_blocks = sum([len(entity[0].split()) for entity in tail_entities if entity[1] == "person"])
    len_date_blocks = sum([len(entity[0].split()) for entity in tail_entities if entity[1] == "date"])
    len_other_blocks = sum([len(entity[0].split()) for entity in tail_entities if entity[1] == "O"])
    # print(num_person_blocks, num_date_blocks, num_other_blocks, len_person_blocks, len_date_blocks, len_other_blocks)
    # print((num_date_blocks == 1 and num_person_blocks == 0 and len_other_blocks <= len_date_blocks))
    # print((num_date_blocks == 0 and num_person_blocks == 1 and len_other_blocks <= len_person_blocks))
    # if there is EXACTLY one person or one date block AND there is <= 1 OTHER block
    # if the OTHER block is not longer than the date or person block
    if num_other_blocks <= 1 and \
        ((num_date_blocks == 1 and num_person_blocks == 0 and len_other_blocks <= len_date_blocks) or
         (num_date_blocks == 0 and num_person_blocks == 1 and len_other_blocks <= len_person_blocks)):
        return head_entities
    else:
        return entities


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
            current_entity_type = "O"
            if current_entity_type == entity_type:
                entity.append(token)
            else:
                if entity:
                    entities.append((' '.join(entity), entity_type))
                    entity = []
                entity_type = current_entity_type
                entity.append(token)
    # Catch any remaining entities
    if entity:
        entities.append((' '.join(entity), entity_type))

    # There is only one task block

    entities_copy = copy.deepcopy(entities)

    if len([entity for entity in entities if entity[1] == "task"]) == 1:
        # for left and right
        entities_copy = strip_date_person_from_right(entities_copy)
        entities_person_date_strip = strip_date_person_from_right(entities_copy[::-1])[::-1]
    else:
        entities_person_date_strip = entities_copy

    corrected_task = " ".join([entity[0].strip(" ") for entity in entities_person_date_strip])

    post_processed_entities = []
    for entity, entity_type in entities:
        if entity_type == "task":
            entity_type = "subtask"
            post_processed_entities.append((entity, entity_type))
        else:
            post_processed_entities.append((entity, entity_type))
    post_processed_entities.append((corrected_task, "task"))

    return post_processed_entities


def join_raw_labels(raw_labels, offset_mapping):
    mask = (offset_mapping[:, :, 0] == 0) & (offset_mapping[:, :, 1] != 0)
    selected_values_list = [raw_labels[i][mask[i]] for i in range(mask.size(0))]
    return selected_values_list


def get_perplexity(outputs, ids2lbl):
    if isinstance(outputs, dict):
        logits = outputs["logits"]
    else:
        logits = outputs
    log_probs = F.log_softmax(logits, dim=-1)
    raw_labels = torch.argmax(logits, dim=-1)

    label_log_probs = log_probs.gather(dim=-1, index=raw_labels.unsqueeze(-1)).squeeze(-1)
    unique_labels = raw_labels.unique()
    perplexities = {}
    for label in unique_labels:
        mask = raw_labels == label
        mean_log_prob = label_log_probs[mask].mean().item()
        perplexities[ids2lbl[label.item()]] = torch.exp(-torch.tensor(mean_log_prob)).item()

    # Overall perplexity
    overall_mean_log_prob = label_log_probs.mean().item()
    overall_perplexity = torch.exp(-torch.tensor(overall_mean_log_prob)).item()
    return perplexities, overall_perplexity


def model_output_to_label_tensor(outputs, offset_mapping, ids2lbl):
    if isinstance(outputs, dict):
        raw_labels = torch.argmax(outputs["logits"], dim=-1)
    else:
        raw_labels = torch.argmax(outputs, dim=-1)
    selected_values_list = join_raw_labels(raw_labels, offset_mapping)
    labels_list = [[ids2lbl[idx.item()] for idx in tensor] for tensor in selected_values_list]
    return labels_list


# def detokenize(tokenized_text):
#     # Remove [CLS] and [SEP] tokens
#     tokenized_text = [t for t in tokenized_text.split() if t not in ['[CLS]', '[SEP]']]
#     # Convert WordPiece tokenization to full words
#     text = ' '.join(tokenized_text).replace(' ##', '')
#     # Handle punctuation and certain characters
#     text = text.replace(" ' s", "'s")  # Handle possessives
#     for punctuation in [' ,', ' .', ' ?', ' !', ' ;', ' :']:
#         text = text.replace(punctuation, punctuation[-1])
#     text = text.replace(" ' re", "'re") # Handle contractions
#     text = text.replace(" / ", "/")  # Remove spaces around slashes
#     return text

def detokenize(tokenized_text):
    # Convert the tokens to text
    text = ''.join(tokenized_text)
    # Remove special tokens
    text = text.replace('[CLS]', '').replace('[SEP]', '')
    # Handle the space prefix "Ġ" character
    text = text.replace('Ġ', ' ').strip()
    # Handle punctuation and certain characters
    text = text.replace(" ' s", "'s")  # Handle possessives
    for punctuation in [' ,', ' .', ' ?', ' !', ' ;', ' :']:
        text = text.replace(punctuation, punctuation[-1])
    text = text.replace(" ' re", "'re")  # Handle contractions
    text = text.replace(" / ", "/")  # Remove spaces around slashes
    return text.strip()  # Remove any leading/trailing spaces


def batch_to_info(batch, tokenizer, ids2lbl, outputs=None) -> List[str]:
    input_ids = batch['input_ids']
    raw_labels = batch['labels'] if 'labels' in batch else None
    offset_mapping = batch['offset_mapping'].squeeze(1)

    # Convert input_ids to tokens
    tok_texts = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
    # Convert tokens to raw_text (by joining tokens and removing special tokens)
    raw_texts = [detokenize(tok_text) for tok_text in tok_texts]

    if outputs is not None:
        # log_probs_all_samples = F.log_softmax(outputs["logits"], dim=-1).cpu().numpy().astype(float).tolist()
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs
        all_predicted_classes = torch.argmax(logits, dim=-1)
        log_probs_all_samples = [None for i in tok_texts]
    else:
        log_probs_all_samples = [None for i in tok_texts]

    # Convert numerical labels and predictions to their string representations
    if raw_labels is None:
        gt_tok_forms = [["None"]]*len(tok_texts)
    else:
        gt_tok_forms = [[ids2lbl[idx.item()] if idx != -100 else "-100" for idx in tensor] for tensor in raw_labels]
    if outputs is not None:
        hyp_tok_forms = [[ids2lbl[idx.item()] for idx in tensor] for tensor in all_predicted_classes]
    else:
        hyp_tok_forms = [["None"]] * len(gt_tok_forms)

    # Get normalized forms
    if raw_labels is None:
        gt_norm_forms = [["None"]] * len(gt_tok_forms)
    else:
        gt_norm_forms = join_raw_labels(raw_labels, offset_mapping)
        gt_norm_forms = [[ids2lbl[idx.item()] for idx in tensor] for tensor in gt_norm_forms]

    if outputs is not None:
        hyp_norm_forms = join_raw_labels(all_predicted_classes, offset_mapping)
        hyp_norm_forms = [[ids2lbl[idx.item()] for idx in tensor] for tensor in hyp_norm_forms]
    else:
        hyp_norm_forms = [["None"]] * len(gt_norm_forms)

    logged_info = []
    for raw_text, tok_text, gt_tok_form, gt_norm_form, hyp_tok_form, hyp_norm_form, log_probs in zip(raw_texts, tok_texts,
                                                                                          gt_tok_forms, gt_norm_forms,
                                                                                          hyp_tok_forms,
                                                                                          hyp_norm_forms, log_probs_all_samples):
        entry = {
            "raw_text": raw_text,
            "tok_text": ' '.join(tok_text),
            "gt_tok_form": ' '.join(gt_tok_form),
            "gt_norm_form": ' '.join(gt_norm_form),
            "hyp_tok_form": ' '.join(hyp_tok_form),
            "hyp_norm_form": ' '.join(hyp_norm_form),
            "log_probs": log_probs
        }
        logged_info.append(entry)

    return logged_info



