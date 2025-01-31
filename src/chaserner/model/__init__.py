import torch
import json
from pathlib import Path
import pytorch_lightning as pl
from transformers import DebertaForTokenClassification
from datasets import load_metric
from chaserner.utils import batch_to_info
import torch.nn as nn
from chaserner.utils.logger import logger

seqeval_metric = load_metric("seqeval")

DEFAULT_WORKING_DIR = Path.home()


class TracingNERModel(torch.nn.Module):
    def __init__(self, model):
        super(TracingNERModel, self).__init__()
        self.model = model
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask).logits

# class DummyNERModel(pl.LightningModule):
#     def __init__(self, num_labels, learning_rate=2e-5):
#         super(DummyNERModel, self).__init__()
#
#         # Simple linear layer
#         self.fc = nn.Linear(768, num_labels)  # Assuming input dimension is 768 (BERT base model size)
#
#         self.learning_rate = learning_rate
#         self.val_outputs = []
#         self.test_outputs = []
#
#     def forward(self, input_ids, attention_mask, labels=None):
#         # Mock a BERT-like output (e.g., 768 features per token)
#         dummy_features = torch.randn(*input_ids.shape[:2], 768)
#
#         # Produce some random outputs
#         output = self.fc(dummy_features)
#
#         # Dummy loss if labels are provided
#         loss = None
#         if labels is not None:
#             loss_fn = nn.CrossEntropyLoss()
#             loss = loss_fn(output.view(-1, output.size(-1)), labels.view(-1))
#
#         return loss, output
#
#     def training_step(self, batch, batch_idx):
#         input_ids = batch['input_ids']
#         labels = batch['labels']
#         loss, _ = self(input_ids, None, labels)
#         return loss
#
#     def on_validation_end(self):
#         print("Model mode after validation:", "training" if self.training else "evaluation")
#
#     def on_validation_start(self):
#         self.eval()
#
#     def on_train_start(self):
#         self.train()  # Set the model to training mode
#
#     def validation_step(self, batch, batch_idx):
#         input_ids = batch['input_ids']
#         labels = batch['labels']
#         loss, _ = self(input_ids, None, labels)
#         self.val_outputs.append({"val_loss": loss})
#
#     def on_validation_epoch_end(self):
#         avg_loss = torch.stack([x['val_loss'] for x in self.val_outputs]).mean()
#         self.log('avg_val_loss', avg_loss)
#         return {'avg_val_loss': avg_loss}
#
#     def test_step(self, batch, batch_idx):
#         self.eval()
#         input_ids = batch['input_ids']
#         labels = batch['labels']
#         loss, logits = self(input_ids, None, labels)
#         all_predicted_classes = torch.argmax(logits, dim=-1)
#         self.test_outputs.append({"test_loss": loss, "labels": labels, "hypotheses": all_predicted_classes})
#
#     def on_test_epoch_end(self):
#         avg_loss = torch.stack([x['test_loss'] for x in self.test_outputs]).mean()
#         self.log('avg_test_loss', avg_loss)
#
#     def configure_optimizers(self):
#         return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

class NERModel(pl.LightningModule):
    def __init__(self, hf_model_name, label_to_id, learning_rate=2e-5, frozen_layers=0, tokenizer=None, working_dir=DEFAULT_WORKING_DIR):
        super(NERModel, self).__init__()
        num_labels = len([k for k in label_to_id.keys() if k not in []])
        self.label_to_id = label_to_id
        self.tokenizer = tokenizer
        self.model = DebertaForTokenClassification.from_pretrained(hf_model_name, num_labels=num_labels)
        self.freeze_encoder_layers(frozen_layers)
        self.learning_rate = learning_rate
        self.val_outputs = []
        self.test_outputs = []
        self.train_epoch_loss = 0.0
        self.train_batch_count = 0
        self.working_dir = Path(working_dir)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def forward_for_tracing(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.train_epoch_loss += loss
        self.train_batch_count += 1
        return loss

    def on_train_epoch_end(self):
        # Compute the average training loss
        avg_train_loss = self.train_epoch_loss / self.train_batch_count
        # Log the average training loss
        self.log('avg_train_loss', avg_train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # Reset the accumulation variables for the next epoch
        self.train_epoch_loss = 0.0
        self.train_batch_count = 0

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     self.log('avg_val_loss', avg_loss)
    #     return {'avg_val_loss': avg_loss}

    def on_validation_start(self):
        self.eval()
        self.val_outputs = []

    def on_train_start(self):
        self.train()  # Set the model to training mode

    def on_test_start(self):
        self.eval()  # Set the model to training mode
        self.test_outputs = []

    def proc_batch_to_lbl_gt_loss(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        raw_labels = batch['labels']
        offset_mapping = batch['offset_mapping']
        outputs = self(input_ids, attention_mask, raw_labels)
        loss = outputs.loss
        logits = outputs["logits"]
        all_predicted_classes = torch.argmax(logits, dim=-1)
        offset_mapping = offset_mapping.squeeze(1)
        mask = (offset_mapping[:, :, 0] == 0) & (offset_mapping[:, :, 1] != 0)
        labels_regrouped = [raw_labels[i][mask[i]] for i in range(mask.size(0))]
        hyps_regrouped = [all_predicted_classes[i][mask[i]] for i in range(mask.size(0))]
        # data_info = batch_to_info(batch, self.tokenizer, {v: k for k, v in self.label_to_id.items()}, outputs=outputs)
        # with open(self.working_dir/'output_test_eval.jsonl', 'a') as f:
        #     f.write('\n'.join([json.dumps(info_sample) for info_sample in data_info]) + "\n")
        return loss, labels_regrouped, hyps_regrouped

    def proc_loss_lbls_hyps_get_metrics(self, loss_lbl_hyps):
        avg_loss = torch.stack([x['test_loss'] for x in loss_lbl_hyps]).mean()
        id2lbl = {v: k for k, v in self.label_to_id.items()}
        try:
            unrolled_lbls = [[id2lbl[s.item()] for s in sample] for batch in loss_lbl_hyps for sample in batch['labels']]
            unrolled_hyps = [[id2lbl[s.item()] for s in sample] for batch in loss_lbl_hyps for sample in batch['hypotheses']]
            seqeval_results = seqeval_metric.compute(predictions=unrolled_hyps, references=unrolled_lbls, mode="strict")
        except ValueError as e:
            print(str(unrolled_hyps))
            print(str(unrolled_lbls))
            raise e
        metrics = {
            "precision": seqeval_results["overall_precision"],
            "recall": seqeval_results["overall_recall"],
            "f1": seqeval_results["overall_f1"],
            "accuracy": seqeval_results["overall_accuracy"],
            'avg_loss': avg_loss,
            "num_samples": len(unrolled_lbls)
        }
        metrics.update({k+"_f1":seqeval_results[k]["f1"] for k in seqeval_results if
         k not in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']})
        metrics.update({k + "_prec": seqeval_results[k]["precision"] for k in seqeval_results if
         k not in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']})
        metrics.update({k + "_rec": seqeval_results[k]["recall"] for k in seqeval_results if
         k not in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']})
        return metrics

    def validation_step(self, batch, batch_idx):
        loss, labels_regrouped, hyps_regrouped = self.proc_batch_to_lbl_gt_loss(batch)
        self.val_outputs.append({"test_loss": loss, "labels": labels_regrouped, "hypotheses": hyps_regrouped})

    def on_validation_epoch_end(self):
        metrics = self.proc_loss_lbls_hyps_get_metrics(self.val_outputs)
        for metric_name, metric_val in metrics.items():
            if metric_name in ["f1", "precision", "recall", "accuracy", "avg_loss"]:
                if isinstance(metric_val, torch.Tensor):
                    metric_val = metric_val.float()  # Ensure it's float32
                else:
                    metric_val = torch.tensor(metric_val, dtype=torch.float32, device=self.device)
                self.log("val_"+metric_name, metric_val, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, labels_regrouped, hyps_regrouped = self.proc_batch_to_lbl_gt_loss(batch)
        self.test_outputs.append({"test_loss": loss, "labels": labels_regrouped, "hypotheses": hyps_regrouped})

    def on_test_epoch_end(self):
        metrics = self.proc_loss_lbls_hyps_get_metrics(self.test_outputs)
        for metric_name, metric_val in metrics.items():
            if isinstance(metric_val, torch.Tensor):
                metric_val = metric_val.float()  # Ensure it's float32
            else:
                metric_val = torch.tensor(metric_val, dtype=torch.float32, device=self.device)
            self.log("test_"+metric_name, metric_val)

    def freeze_encoder_layers(self, num_layers_to_freeze=6):
        """
        Freeze the first `num_layers_to_freeze` of the BERT model.
        """
        for layer in self.model.deberta.encoder.layer[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

