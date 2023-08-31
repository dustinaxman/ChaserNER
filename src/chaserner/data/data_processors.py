from datasets import load_dataset
import torch
import json
import random
from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from chaserner.data.simulator import simulate_train_dev_test
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from chaserner.utils.logger import logger
from chaserner.utils import batch_to_info
import multiprocessing

NUM_WORKERS = 0#multiprocessing.cpu_count()


class NERDataset(Dataset):
    def __init__(self, data, label_to_id, tokenizer_name='SpanBERT/spanbert-base-cased', max_length=512):
        self.data = data
        self.label_to_id = label_to_id
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.all_data_info = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if len(self.data[index]) > 1:
            text, _, ner_label_strings = self.data[index]
        else:
            text = self.data[index]

        # Tokenizing the text
        tokenized_data = self.tokenizer(
            text.split(),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            is_split_into_words=True,
            return_offsets_mapping=True
        )

        if len(self.data[index]) > 1:
            offsets = tokenized_data['offset_mapping'][0]

            # Initialize token labels with a special label (e.g., -100 which is ignored by Hugging Face's models)
            token_labels = [-100] * self.max_length

            label_idx = 0
            for token_idx, (start, end) in enumerate(offsets):
                # Check if this token corresponds to a new word
                if start == 0 and label_idx < len(ner_label_strings) and end != 0:
                    token_labels[token_idx] = self.label_to_id[ner_label_strings[label_idx]]
                    label_idx += 1
        else:
            token_labels = [-100] * self.max_length

        batch = {
            'input_ids': tokenized_data['input_ids'],
            'attention_mask': tokenized_data['attention_mask'],
            'labels': torch.tensor(token_labels, dtype=torch.long).unsqueeze(0),
            'offset_mapping': tokenized_data['offset_mapping']
        }

        data_info = batch_to_info(batch, self.tokenizer, {v: k for k, v in self.label_to_id.items()})
        self.all_data_info.extend(data_info)
        # Save the jsonl_data to a file (if needed)
        return {
            'input_ids': tokenized_data['input_ids'].squeeze(),
            'attention_mask': tokenized_data['attention_mask'].squeeze(),
            'labels': torch.tensor(token_labels, dtype=torch.long),
            'offset_mapping': tokenized_data['offset_mapping']
        }


class SimulatorNERDataModule(LightningDataModule):
    def __init__(self, batch_size=32, tokenizer_name='SpanBERT/spanbert-base-cased', max_length=512, config_path='/tmp/config.json'):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length

        # Loading data
        train, dev, test = simulate_train_dev_test()

        all_data = train + dev + test

        random.shuffle(all_data)

        def proc_dict(dict_val):
            return " | ".join([k+":"+v for k, v in dict_val.items()])

        with open("/Users/deaxman/Downloads/simulated_data.txt", "w") as f:
            #f.write("\n".join(["\t".join([" ".join([txt+"|"+lbl for txt, lbl in zip(txt_input.split(), raw_labels)]), proc_dict(labels)]) for txt_input, labels, raw_labels in all_data]))
            f.write("\n".join(["\t".join(
                [txt_input, proc_dict(labels), " ".join(raw_labels)]) for
                               txt_input, labels, raw_labels in all_data]))

        all_labels = [label for _, _, sample_labels in train for label in sample_labels]
        unique_labels = set(all_labels)
        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        # self.label_to_id["[SEP]"] = len(self.label_to_id)
        # self.label_to_id["[CLS]"] = len(self.label_to_id)
        # self.label_to_id["-100"] = -100  # Typically, this label is used to ignore tokens.

        config = {}
        config["lbl2ids"] = self.label_to_id
        config["max_length"] = max_length

        with open(config_path, "w") as f:
            json.dump(config, f)

        # Creating datasets
        self.train_dataset = NERDataset(train, self.label_to_id, tokenizer_name=self.tokenizer_name,
                                        max_length=self.max_length)

        for sample in self.train_dataset:
            pass
        with open(Path.home()/'Downloads/output_test.jsonl', 'w') as f:
            f.write('\n'.join([json.dumps(info_sample) for info_sample in self.train_dataset.all_data_info]) + "\n")
        self.val_dataset = NERDataset(dev, self.label_to_id, tokenizer_name=self.tokenizer_name,
                                      max_length=self.max_length)
        self.test_dataset = NERDataset(test, self.label_to_id, tokenizer_name=self.tokenizer_name,
                                       max_length=self.max_length)
        logstr = " ".join([str(len(self.train_dataset)), str(len(self.val_dataset)), str(len(self.test_dataset))])
        logger.info(logstr)

    def setup(self, stage=None):
        pass
        # # Loading data
        # train, dev, test = simulate_train_dev_test()
        #
        # all_labels = [label for _, _, sample_labels in train for label in sample_labels]
        # unique_labels = set(all_labels)
        # self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        # #TODO save label to id
        #
        # # Creating datasets
        # self.train_dataset = NERDataset(train, self.label_to_id, tokenizer_name=self.tokenizer_name, max_length=self.max_length)
        # self.val_dataset = NERDataset(dev, self.label_to_id, tokenizer_name=self.tokenizer_name, max_length=self.max_length)
        # self.test_dataset = NERDataset(test, self.label_to_id, tokenizer_name=self.tokenizer_name, max_length=self.max_length)
        # logstr = " ".join([str(len(self.train_dataset)), str(len(self.val_dataset)), str(len(self.test_dataset))])
        # logger.info(logstr)

    def train_dataloader(self):
        logger.info(f"Creating DataLoader with batch size: {self.batch_size}")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)








# Load the CoNLL-2003 dataset
# dataset = load_dataset("conll2003")


# # Tokenization
# from transformers import BertTokenizerFast
#
# tokenizer = BertTokenizerFast.from_pretrained('SpanBERT/spanbert-base-cased')
# def encode_examples(example):
#     # Tokenize inputs and labels
#     tokenized_inputs = tokenizer(example['tokens'], truncation=True, padding='max_length', is_split_into_words=True)
#     labels = example['ner_tags']
#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs
#
# dataset = dataset.map(encode_examples)

# num_labels = dataset["train"].features["ner_tags"].feature.num_classes
#
# model = NERModel(num_labels=num_labels)
#
# trainer = Trainer(max_epochs=3)
# trainer.fit(model, dataset["train"], dataset["validation"])