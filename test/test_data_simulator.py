import pytest
from chaserner.data.simulator import simulate_train_dev_test


def is_valid_sample(sample):
    """Utility function to check the validity of a sample."""
    # Check if sample is a tuple of length 3
    if not isinstance(sample, tuple) or len(sample) != 3:
        return False

    text, label_entity_map, ner_labels = sample

    # Check if text is a string
    if not isinstance(text, str):
        return False

    # Check if label_entity_map is a dictionary
    if not isinstance(label_entity_map, dict):
        return False

    # Check if ner_labels is a list
    if not isinstance(ner_labels, list):
        return False

    # Ensure the number of space-separated tokens in the text string
    # matches the length of the NER labels list
    if len(text.split()) != len(ner_labels):
        return False

    return True


def test_simulate_data_structure():
    train, dev, test = simulate_train_dev_test()
    all_data = train + dev + test

    # Check if all_data is a list
    assert isinstance(all_data, list), "all_data should be a list"

    # Check each sample in all_data
    for sample in all_data:
        assert is_valid_sample(sample), f"Invalid sample: {sample}"


def test_dataset_splits():
    train, dev, test = simulate_train_dev_test()

    # Ensure that the data is split into train, dev, and test sets
    assert isinstance(train, list), "train should be a list"
    assert isinstance(dev, list), "dev should be a list"
    assert isinstance(test, list), "test should be a list"


def is_entity_present_in_text(entity, text):
    """Utility function to check if an entity is present in the text."""
    return entity in text

def test_entities_in_text():
    train, dev, test = simulate_train_dev_test()
    all_data = train + dev + test

    for sample in all_data:
        text, label_entity_map, _ = sample
        for entity in label_entity_map.values():
            assert is_entity_present_in_text(entity, text), f"Entity '{entity}' not found in text: {text}"