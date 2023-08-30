from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from chaserner.data.data_processors import SimulatorNERDataModule
from chaserner.model import NERModel#, DummyNERModel
from datetime import datetime
from pathlib import Path

save_model_dir = Path().home()/"Downloads/saved_model/"

early_stop_callback = EarlyStopping(
    monitor='val_avg_loss',  # Monitor the average validation loss
    min_delta=0.00,
    patience=3,
    verbose=True,
    mode='min'
)

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# Define model checkpoint criteria
checkpoint_callback = ModelCheckpoint(
    monitor='val_avg_loss',
    dirpath='/Users/deaxman/Downloads/checkpoints',
    filename=f'ner-{current_time}-epoch{{epoch:02d}}-val_avg_loss{{val_avg_loss:.2f}}',
    save_top_k=1,
    mode='min'
)

# Initialize trainer with callbacks
trainer = Trainer(
    accelerator="mps",
    max_epochs=10,
    callbacks=[early_stop_callback, checkpoint_callback]
)


ner_data_module = SimulatorNERDataModule(batch_size=128, tokenizer_name='SpanBERT/spanbert-base-cased', max_length=32, config_path=save_model_dir/"config.json")

ner_data_module.setup('fit')

num_labels = len([k for k in ner_data_module.label_to_id.keys() if k not in []])

#model = DummyNERModel(num_labels=num_labels)
model = NERModel(lbl2id=ner_data_module.label_to_id, learning_rate=2e-5, frozen_layers=4)

trainer.fit(model, ner_data_module)

trainer.validate(datamodule=ner_data_module)

trainer.test(datamodule=ner_data_module)
