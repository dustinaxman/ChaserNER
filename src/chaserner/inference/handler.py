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

# class NERModelHandlerWithSelfBatching(BaseHandler):
#
#     def __init__(self):
#         def __init__(self):
#             self.model = None
#
#         self.tokenizer = None
#         self.device = None
#         self.queue = queue.Queue()
#         self.lock = threading.Lock()
#         self.initialized = False
#         self.timer_started = False
#         self.start_time = None
#         # Dictionary to store threading events for each request
#         self.wait_events = {}
#
#         # Dictionary to store batch results
#         self.all_results = {}
#
#         # Thread to handle batching logic
#         self.batch_thread = threading.Thread(target=self.batch_processor)
#         self.batch_thread.start()
#
#     def initialize(self, ctx):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # Load model from checkpoint
#         model_dir = ctx.system_properties.get("model_dir")
#         self.model = BertForTokenClassification.from_pretrained(model_dir)
#         self.model.to(self.device)
#         self.model.eval()
#
#         # Initialize the BERT tokenizer
#         self.tokenizer = BertTokenizer.from_pretrained(model_dir)
#         self.initialized = True
#
#     def preprocess(self, data):
#         text = data[0].get("data")
#         if text is None:
#             text = data[0].get("body")
#         return text
#
#     def batch_processor(self):
#         while True:
#             # Check if we should process a batch
#             if self.queue.qsize() >= MAX_BATCH_SIZE or (
#                     self.timer_started and time.time() - self.start_time > WAIT_TIME):
#                 with self.lock:
#                     # Reset timer
#                     self.timer_started = False
#                     self.start_time = None
#
#                     # Collect batch data
#                     batch_data = []
#                     while not self.queue.empty() and len(batch_data) < MAX_BATCH_SIZE:
#                         batch_data.append(self.queue.get())
#
#                     # Extract texts and request ids
#                     texts = [item["text"] for item in batch_data]
#                     request_ids = [item["request_id"] for item in batch_data]
#
#                     # Process the batch (e.g., model inference)
#                     results = self.process_batch(texts)  # Placeholder for your batch processing
#
#                     # Store results
#                     for request_id, result in zip(request_ids, results):
#                         self.all_results[request_id] = result
#
#                     # Notify waiting handle functions
#                     for request_id in request_ids:
#                         self.wait_events[request_id].set()
#
#             time.sleep(0.01)
#
#     def handle(self, data, context):
#         # ... preprocess ...
#
#         request_id = context.request_id  # Assuming each request has a unique ID
#         self.wait_events[request_id] = threading.Event()
#
#         # Add to queue for batch processing
#         self.queue.put({"text": text, "request_id": request_id})
#
#         # Wait for result to be available
#         self.wait_events[request_id].wait()
#
#         # Fetch result
#         result = self.all_results.pop(request_id, None)
#
#         # Clean up
#         del self.wait_events[request_id]
#
#         return result






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
        text_input_list_for_inference = [text_input for text_input in input_text_list if text_input not in self.lfucache.cache]
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
        return [self.lfucache.get(text_input) if text_input in self.lfucache.cache else temp_cache[text_input] for text_input in input_text_list]



_service = NERModelHandler()


def handle(data_batch, context):
    if not _service.initialized:
        _service.initialize(context)
    if data_batch is None:
        return None

    input_text_list = _service.preprocess(data_batch)
    result = _service.inference(input_text_list)
    #result = _service.postprocess(result)
    return result

