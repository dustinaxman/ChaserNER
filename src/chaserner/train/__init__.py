from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from chaserner.data.data_processors import SimulatorNERDataModule
from chaserner.model import NERModel#, DummyNERModel
from datetime import datetime
from pathlib import Path
import json
import shutil



early_stop_callback = EarlyStopping(
    monitor='val_avg_loss',  # Monitor the average validation loss
    min_delta=0.00,
    patience=2,
    verbose=True,
    mode='min'
)

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

save_model_dir = Path().home()/f"Downloads/saved_model_{current_time}/"

if not save_model_dir.exists():
    save_model_dir.mkdir(parents=True)



config_path = save_model_dir / "config.json"

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
    max_epochs=5,
    callbacks=[early_stop_callback, checkpoint_callback]
)
# TODO: try bigger model
# TODO: add in more data (bigger ratio)
# tokenizer_name = 'SpanBERT/spanbert-base-cased'
# hf_model_name = 'SpanBERT/spanbert-base-cased'
hf_model_name = "microsoft/deberta-base"
tokenizer_name = "microsoft/deberta-base"

ner_data_module = SimulatorNERDataModule(batch_size=64, tokenizer_name=tokenizer_name, max_length=64, config_path=config_path)

ner_data_module.setup('fit')

num_labels = len([k for k in ner_data_module.label_to_id.keys() if k not in []])

#model = DummyNERModel(num_labels=num_labels)
model = NERModel(hf_model_name=hf_model_name, label_to_id=ner_data_module.label_to_id, learning_rate=2e-5, frozen_layers=0, tokenizer=ner_data_module.train_dataset.tokenizer)

trainer.fit(model, ner_data_module)

best_checkpoint = Path(checkpoint_callback.best_model_path)

destination = save_model_dir / best_checkpoint.name
shutil.copy(best_checkpoint, destination)

#Update the config with the best checkpoint path
with config_path.open() as f:
    config = json.load(f)

config["best_checkpoint"] = best_checkpoint.name
config["tokenizer_name"] = tokenizer_name
config["hf_model_name"] = hf_model_name

with config_path.open("w") as f:
    json.dump(config, f)



trainer.validate(datamodule=ner_data_module)

trainer.test(datamodule=ner_data_module)

