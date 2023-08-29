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


def merge_subwords(tokens):
    merged_tokens = []
    for token in tokens:
        # If it's a subword token
        if token.startswith("##"):
            if merged_tokens:
                # Merge with the previous token
                merged_tokens[-1] = merged_tokens[-1] + token[2:]
        else:
            merged_tokens.append(token)
    return merged_tokens



# # tokenized_output = tokenizer("ChatGPT is developed by OpenAI.", return_tensors="pt")
# tokens = tokenizer.convert_ids_to_tokens(tokenized_output["input_ids"][0])
# merged_tokens = merge_subwords(tokens)



# tokenized_data = self.tokenizer(
#             text.split(),
#             padding='max_length',
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors='pt',
#             is_split_into_words=True,
#             return_offsets_mapping=True
#         )

def join_raw_labels(raw_labels, offset_mapping):
    mask = (offset_mapping[:, :, 0] == 0) & (offset_mapping[:, :, 1] != 0)
    selected_values_list = [raw_labels[i][mask[i]] for i in range(mask.size(0))]
    return selected_values_list


def model_output_to_label_tensor(outputs, offset_mapping, ids2lbl):
    raw_labels = torch.argmax(outputs.logits, dim=-1)
    selected_values_list = join_raw_labels(raw_labels, offset_mapping)
    labels_list = [[ids2lbl[idx.item()] for idx in tensor] for tensor in selected_values_list]
    return labels_list



def input_text_list_to_extracted_entities(input_text_list, model_path, config_path):
    tokenizer_name = 'SpanBERT/spanbert-base-cased'
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    with open(config_path) as f:
        config = json.load(f)
    ids2lbl = {v: k for k, v in config["lbl2ids"].items()}
    max_length = config["max_length"]
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
    return [{k: v for v, k in extract_entities(input_text.split(), labels)} for input_text, labels in zip(input_text_list, labels_list)]


#
# offsets = tokenized_data['offset_mapping'][0]
#
# # Initialize token labels with a special label (e.g., -100 which is ignored by Hugging Face's models)
# token_labels = [-100] * self.max_length
#
# label_idx = 0
# for token_idx, (start, end) in enumerate(offsets):
#     # Check if this token corresponds to a new word
#     if start == 0 and label_idx < len(ner_label_strings)  and end != 0:
#         token_labels[token_idx] = self.label_to_id[ner_label_strings[label_idx]]
#         label_idx += 1

#only take the labels for the non subwords! exactly condition above



