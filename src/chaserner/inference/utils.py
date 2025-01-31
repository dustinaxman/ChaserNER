import torch
# import torch.quantization
# from torch.quantization import quantize_dynamic
from transformers import DebertaTokenizerFast
import json
from chaserner.model import NERModel
from chaserner.utils import model_output_to_label_tensor, extract_entities, batch_to_info, process_entities_to_dict, get_perplexity
from pathlib import Path
import time
import pprint
import re

#torch.backends.quantized.engine = 'qnnpack'

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
    tokenizer = DebertaTokenizerFast.from_pretrained(tokenizer_name, add_prefix_space=True)
    # TODO: remove the extra args here later!!! for later models
    if "torchscript_model" in config:
        model_path = config_path.parent / config["torchscript_model"]
        print("LOADING TORCHSCRIPT FILE")
        model = torch.jit.load(str(model_path), map_location=device)#, strict=False)
    else:
        print("LOADING MODEL CHECKPOINT")
        model = NERModel.load_from_checkpoint(checkpoint_path=model_path, hf_model_name=tokenizer_name,
                                              label_to_id=config["lbl2ids"], map_location=device, strict=False)
    model.eval()
    model = model.to(device)
    #quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return model, tokenizer, max_length, ids2lbl



pattern = re.compile(r'<[A-Z0-9]+>')

def pre_proc_input_text_list(input_text_list):
    output_text_list = []
    replacements_list = []
    for input_text in input_text_list:
        replacements = {}
        name_index = 0
        def replace(match):
            nonlocal name_index
            placeholder = match.group(0)
            if placeholder not in replacements:
                replacements[placeholder] = "dustin_" + str(name_index)#stand_in_names[name_index]
                name_index += 1
            return replacements[placeholder]

        result_string = pattern.sub(replace, input_text)
        output_text_list.append(result_string)
        replacements_list.append(replacements)
    return output_text_list, replacements_list


def proc_output_entities(entity_label_pairs, replacements):
    replacements_undo_dict = {v: k for k, v in replacements.items()}
    entity_label_pairs_proc = []
    for text, label in entity_label_pairs:
        # add code to check if any of the keys in replacements_undo_dict exist in the "text" string variable.
        # If so, replace the substring of the key with the value for that key (also a string) and set it to text_proc
        text_proc = text
        for key, value in replacements_undo_dict.items():
            if key in text_proc:
                text_proc = text_proc.replace(key, value)
        entity_label_pairs_proc.append((text_proc, label))
    return entity_label_pairs_proc



def is_valid_slack_userid(string):
    pattern = r'^<U[A-Z0-9]+>$'
    return bool(re.match(pattern, string))

def filter_extracted_entities(entity_extracted_samples):
    entity_extracted_samples_filtered = entity_extracted_samples.copy()  # Create a copy to avoid modifying the original
    for sample in entity_extracted_samples_filtered:
        sample["extracted_entities"]["person"] = [
            p for p in sample["extracted_entities"]["person"] if is_valid_slack_userid(p)
        ]
    return entity_extracted_samples_filtered

def run_ner_model(input_text_list, model, tokenizer, max_length, ids2lbl, device, info_log=None):
    start_time = time.time()
    input_text_list = [txt.strip() for txt in input_text_list]
    input_text_list_proc, replacements_list = pre_proc_input_text_list(input_text_list)
    token_lengths = [len(tokenizer.encode(txt, add_special_tokens=True)) for txt in input_text_list]
    # Find the maximum token length from the tokenized texts
    max_input_length = max(token_lengths)

    max_length = min(max_input_length, max_length)
    tokenized_data = tokenizer(
        [txt.split() for txt in input_text_list_proc],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
        return_tensors='pt',
        return_offsets_mapping=True
    ).to(device)
    total_time = time.time() - start_time
    print(f"TOKENIZER DONE: {total_time}")
    print("RUNNING MODEL")
    with torch.inference_mode():
        outputs = model(tokenized_data["input_ids"], tokenized_data["attention_mask"])
    total_time = time.time() - start_time
    print(f"MODEL DONE: {total_time}")
    # print(outputs)
    labels_list = model_output_to_label_tensor(outputs, tokenized_data["offset_mapping"], ids2lbl)

    # perplexity_by_label, overall_perplexity = get_perplexity(outputs, ids2lbl)
    # print("perplexity:")
    # print(overall_perplexity)
    # print(perplexity_by_label)

    entity_extracted_samples = [{"input_text": input_text,
                                 "extracted_entities": process_entities_to_dict(proc_output_entities(extract_entities(input_text_proc.split(), labels), replacements))}
                                for input_text, input_text_proc, labels, replacements in zip(input_text_list, input_text_list_proc, labels_list, replacements_list)]
    total_time = time.time() - start_time
    print(f"ALL: {total_time}")
    entity_extracted_samples = filter_extracted_entities(entity_extracted_samples)
    if info_log:
        return entity_extracted_samples, {"model_input_and_output": batch_to_info(tokenized_data,
                                                                                  tokenizer,
                                                                                  ids2lbl,
                                                                                  outputs=outputs),
                                          "model_runtime": total_time}
    else:
        return entity_extracted_samples


def input_text_list_to_extracted_entities(input_text_list, config_path, device):
    start_time_1 = time.time()
    config_path = Path(config_path)
    model, tokenizer, max_length, ids2lbl = load_model(config_path, device)
    print(f"Using device: {device}")

    #max_input_length = max([len(txt.split()) for txt in input_text_list])

    start_time_2 = time.time()

    entity_extracted_samples, info_dict = run_ner_model(input_text_list, model, tokenizer, max_length, ids2lbl, device, info_log=True)
    # for tok in info_dict["model_input_and_output"][0]["log_probs"]:
    #     print(tok.tolist())
    pprint.pprint(info_dict)
    #print(ids2lbl)
    total_time = time.time() - start_time_2
    load_time = start_time_2 - start_time_1
    per_utt_time = total_time/float(len(input_text_list))
    print(f"Total time: {total_time}\nPer utt time: {per_utt_time}\nTotal utts: {len(input_text_list)}\nLoad time: {load_time}")
    # for input_text, labels in zip(input_text_list, labels_list):
    #     print(input_text)
    #     print(labels)
    return entity_extracted_samples


