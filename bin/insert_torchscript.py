from pathlib import Path
import argparse
import json
import torch
from transformers import DebertaTokenizerFast
from chaserner.model import NERModel

EXAMPLE_TEXT = "this is a team effort dustin axman this is really import, please work very hard to finish the report on the operating costs of black mountain economy flies in the rainforest by friday the twenty seventh this is a team effort dustin axman this is really import, please work very hard to finish the report on the operating costs of black mountain economy flies in the rainforest by friday the twenty seventh"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="insert torchscript model.")

    parser.add_argument('--config_path', type=str, required=True,
                        help="Path to the config.")

    args = parser.parse_args()

    config_path = Path(args.config_path)
    with open(config_path) as f:
        config = json.load(f)

    max_length = config["max_length"]
    ids2lbl = {v: k for k, v in config["lbl2ids"].items()}
    model_path = config_path.parent/config["best_checkpoint"]
    tokenizer_name = config["tokenizer_name"]
    tokenizer = DebertaTokenizerFast.from_pretrained(tokenizer_name, add_prefix_space=True)
    # TODO: remove the extra args here later!!! for later models
    model = NERModel.load_from_checkpoint(checkpoint_path=model_path, hf_model_name=tokenizer_name, label_to_id=config["lbl2ids"], map_location="cpu", strict=False)
    model.eval()
    model = model.to('cpu')
    tokenized_data = tokenizer(
        [txt.split() for txt in [EXAMPLE_TEXT]*2],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
        is_split_into_words=True,
        return_offsets_mapping=True
    ).to("cpu")
    example_inputs = (tokenized_data["input_ids"], tokenized_data["attention_mask"])
    traced_model = model.to_torchscript(method='trace', example_inputs=example_inputs, strict=False)
    #traced_model = torch.jit.trace(model, example_inputs)
    torchscript_model_path = config_path.parent/'model.pt'
    traced_model.save(torchscript_model_path)
    config["torchscript_model"] = str(torchscript_model_path.name)
    with open(config_path, "w") as f:
        json.dump(config, f)
