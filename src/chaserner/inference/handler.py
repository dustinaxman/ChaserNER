import torch
import threading
import time
import queue
import json
from pathlib import Path
from chaserner.utils import LFUCache
from chaserner.inference.utils import load_model, run_ner_model
import time

MAX_BATCH_SIZE = 32
WAIT_TIME = 0.5
SEQ_LENGTH = 64
CACHE_SEQ_LEN_THRESH = SEQ_LENGTH * 15
LFU_CACHE_CAPACITY = 100000


class NERModelHandler():
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.initialized = False
        self.lfucache = LFUCache(LFU_CACHE_CAPACITY)
        # TODO: backup and load the lfu cache to keep it synced across all workers

    def initialize(self, context):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_directory = Path(context.system_properties['model_dir'])
        config_path = model_directory/'config.json'
        self.model, self.tokenizer, self.max_length, self.ids2lbl = load_model(config_path, self.device)
        self.initialized = True

    def preprocess(self, data_batch):
        print(f"data_batch: {data_batch}")
        return [data["body"]["text"] for data in data_batch]

    def inference(self, input_text_list):
        # The list input can have items that are either lists of strings or strings, for backward compatibility
        assert(all([isinstance(call_textfield_input, str) or isinstance(call_textfield_input, list) for call_textfield_input in input_text_list]))
        all_list_form_textfield_input = [[call_textfield_input] if isinstance(call_textfield_input, str) else call_textfield_input for call_textfield_input in input_text_list]
        input_text_list_processed = [text_elem for list_of_text_elem in all_list_form_textfield_input for text_elem in list_of_text_elem]
        text_input_list_for_inference = [text_input for text_input in input_text_list_processed if
                                         text_input not in self.lfucache.cache]
        if len(text_input_list_for_inference) > 0:
            print(self.device)
            print("LEN INFERENCE TEXT", len(text_input_list_for_inference))
            print(text_input_list_for_inference)
            starttime = time.time()
            output_samples = run_ner_model(text_input_list_for_inference, self.model, self.tokenizer, self.max_length, self.ids2lbl, self.device)
            print(time.time() - starttime)
            entity_extracted_samples = [sample["extracted_entities"] for sample in output_samples]
            temp_cache = {}
            for text_input, extracted_entity_dict in zip(text_input_list_for_inference, entity_extracted_samples):
                if len(text_input) < CACHE_SEQ_LEN_THRESH:
                    self.lfucache.put(text_input, extracted_entity_dict)
                else:
                    temp_cache[text_input] = extracted_entity_dict
        return [[self.lfucache.get(text_input) if text_input in self.lfucache.cache else temp_cache[text_input] for text_input in list_of_text_elem] for list_of_text_elem in all_list_form_textfield_input]



_service = NERModelHandler()


def handle(data_batch, context):
    if not _service.initialized:
        print("Initializing service")
        _service.initialize(context)
    if data_batch is None:
        return None
    starttime = time.time()
    input_text_list = _service.preprocess(data_batch)
    preproc_time = time.time() - starttime
    print(f"PREPROCESS TIME: {preproc_time}")
    result = _service.inference(input_text_list)
    preproc_time = time.time() - starttime
    print(f"POST INFERENCE TIME: {preproc_time}")
    #result = _service.postprocess(result)
    return result

