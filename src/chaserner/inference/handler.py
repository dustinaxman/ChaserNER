import torch
from transformers import BertTokenizer, BertForTokenClassification
from ts.torch_handler.base_handler import BaseHandler
import threading
import time
import queue

MAX_BATCH_SIZE = 32
WAIT_TIME = 0.5


class BERTNERHandler(BaseHandler):

    def __init__(self):
        def __init__(self):
            self.model = None

        self.tokenizer = None
        self.device = None
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.initialized = False
        self.timer_started = False
        self.start_time = None
        # Dictionary to store threading events for each request
        self.wait_events = {}

        # Dictionary to store batch results
        self.all_results = {}

        # Thread to handle batching logic
        self.batch_thread = threading.Thread(target=self.batch_processor)
        self.batch_thread.start()

    def initialize(self, ctx):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model from checkpoint
        model_dir = ctx.system_properties.get("model_dir")
        self.model = BertForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        return text

    def batch_processor(self):
        while True:
            # Check if we should process a batch
            if self.queue.qsize() >= MAX_BATCH_SIZE or (
                    self.timer_started and time.time() - self.start_time > WAIT_TIME):
                with self.lock:
                    # Reset timer
                    self.timer_started = False
                    self.start_time = None

                    # Collect batch data
                    batch_data = []
                    while not self.queue.empty() and len(batch_data) < MAX_BATCH_SIZE:
                        batch_data.append(self.queue.get())

                    # Extract texts and request ids
                    texts = [item["text"] for item in batch_data]
                    request_ids = [item["request_id"] for item in batch_data]

                    # Process the batch (e.g., model inference)
                    results = self.process_batch(texts)  # Placeholder for your batch processing

                    # Store results
                    for request_id, result in zip(request_ids, results):
                        self.all_results[request_id] = result

                    # Notify waiting handle functions
                    for request_id in request_ids:
                        self.wait_events[request_id].set()

            time.sleep(0.01)

    def handle(self, data, context):
        # ... preprocess ...

        request_id = context.request_id  # Assuming each request has a unique ID
        self.wait_events[request_id] = threading.Event()

        # Add to queue for batch processing
        self.queue.put({"text": text, "request_id": request_id})

        # Wait for result to be available
        self.wait_events[request_id].wait()

        # Fetch result
        result = self.all_results.pop(request_id, None)

        # Clean up
        del self.wait_events[request_id]

        return result


_service = BERTNERHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
    if data is None:
        return None

    # Extract the endpoint (or identifier) from data or context
    endpoint = data[0].get("endpoint", "default_endpoint")  # Adjust as needed

    text = _service.preprocess(data)
    result = _service.inference(text, endpoint)
    result = _service.postprocess(result)
    return result

