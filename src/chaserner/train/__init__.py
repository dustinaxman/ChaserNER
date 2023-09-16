from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from chaserner.data.data_processors import SimulatorNERDataModule
from chaserner.model import NERModel#, DummyNERModel
from datetime import datetime
from pathlib import Path
import json
import shutil
from chaserner.utils.logger import logger

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')



def train_and_save_model(save_model_dir, tokenizer_name = 'SpanBERT/spanbert-base-cased', hf_model_name = 'SpanBERT/spanbert-base-cased', max_epochs=5, batch_size=128, max_length=64, learning_rate=2e-5, frozen_layers=0, min_delta=0.00, patience=2):
    save_model_dir = Path(save_model_dir)
    config_path = save_model_dir / "config.json"

    if not save_model_dir.exists():
        save_model_dir.mkdir(parents=True)

    early_stop_callback = EarlyStopping(
        monitor='val_avg_loss',  # Monitor the average validation loss
        min_delta=min_delta,
        patience=patience,
        verbose=True,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_avg_loss',
        dirpath=save_model_dir,
        filename=f'ner-{current_time}-epoch{{epoch:02d}}-val_avg_loss{{val_avg_loss:.2f}}',
        save_top_k=1,
        mode='min'
    )

    trainer = Trainer(
        accelerator="auto",
        max_epochs=max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback]
    )



    ner_data_module = SimulatorNERDataModule(batch_size=batch_size, tokenizer_name=tokenizer_name, max_length=max_length, config_path=config_path)
    ner_data_module.setup('fit')
    num_labels = len([k for k in ner_data_module.label_to_id.keys() if k not in []])
    #model = DummyNERModel(num_labels=num_labels)
    model = NERModel(hf_model_name=hf_model_name, label_to_id=ner_data_module.label_to_id, learning_rate=learning_rate, frozen_layers=frozen_layers, tokenizer=ner_data_module.train_dataset.tokenizer)
    trainer.fit(model, ner_data_module)

    best_checkpoint = Path(checkpoint_callback.best_model_path)

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


    logger.info(f"Model saved to: {save_model_dir}")
    return save_model_dir