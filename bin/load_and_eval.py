from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from chaserner.data.data_processors import SimulatorNERDataModule
from chaserner.model import NERModel, DummyNERModel

early_stop_callback = EarlyStopping(
    monitor='avg_val_loss',  # Monitor the average validation loss
    min_delta=0.00,
    patience=3,
    verbose=True,
    mode='min'
)

# Define model checkpoint criteria
checkpoint_callback = ModelCheckpoint(
    monitor='avg_val_loss',
    dirpath='/Users/deaxman/Downloads/checkpoints',
    filename='ner-epoch{epoch:02d}-avg_val_loss{avg_val_loss:.2f}',
    save_top_k=1,
    mode='min'
)

# Initialize trainer with callbacks
trainer = Trainer(
    accelerator="mps",
    max_epochs=5,
    callbacks=[early_stop_callback, checkpoint_callback]
)


ner_data_module = SimulatorNERDataModule(batch_size=128, tokenizer_name='SpanBERT/spanbert-base-cased', max_length=32)


ner_data_module.setup('fit')


num_labels = len(ner_data_module.label_to_id)


model = NERModel.load_from_checkpoint(checkpoint_path=Path("~/Downloads/checkpoints/ner-2023-08-28_15-18-29-epochepoch=04-avg_val_lossavg_val_loss=0.18.ckpt"), num_labels=num_labels)


trainer.validate(model, datamodule=ner_data_module)


trainer.test(model, datamodule=ner_data_module)

