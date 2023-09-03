import pytest
from unittest.mock import MagicMock, patch
from your_handler_module import NERModelHandler, handle

# Sample mock data
MOCK_DATA_BATCH = [{"text": "sample text 1"}, {"text": "sample text 2"}]
MOCK_CONFIG = {
    "config.json": "/path/to/config"
}

# Mocking external dependencies
@patch("your_handler_module.load_model")
@patch("your_handler_module.run_ner_model")
def test_handler(mock_run_ner_model, mock_load_model):
    # Mocking the outputs of external dependencies
    mock_load_model.return_value = (MagicMock(), MagicMock(), 128, {1: "label"})
    mock_run_ner_model.return_value = [{"extracted_entities": {"label": [1, 2]}} for _ in MOCK_DATA_BATCH]

    # Testing the initialization
    context = MagicMock()
    context.artifacts = MOCK_CONFIG
    handler_instance = NERModelHandler()
    handler_instance.initialize(context)
    assert handler_instance.initialized

    # Testing preprocessing
    preprocessed_data = handler_instance.preprocess(MOCK_DATA_BATCH)
    assert preprocessed_data == ["sample text 1", "sample text 2"]

    # Testing inference
    inference_results = handler_instance.inference(preprocessed_data)
    assert inference_results == [{"label": [1, 2]}, {"label": [1, 2]}]

    # Testing the handle function
    handle_results = handle(MOCK_DATA_BATCH, context)
    assert handle_results == [{"label": [1, 2]}, {"label": [1, 2]}]

# This is how you run the test:
# pytest your_test_module.py
