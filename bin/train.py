from datetime import datetime
from pathlib import Path
import argparse
from chaserner.train import train_and_save_model


def load_args():
    parser = argparse.ArgumentParser(description="Train the ChaserNER model for NER")

    parser.add_argument("--save_model_dir",
                        help="Directory to save the model")
    parser.add_argument("--tokenizer_name", type=str, default="SpanBERT/spanbert-base-cased",
                        help="Name or path to the tokenizer")
    parser.add_argument("--hf_model_name", type=str, default="SpanBERT/spanbert-base-cased",
                        help="Name or path to the HuggingFace model")
    parser.add_argument("--max_epochs", type=int, default=5,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--max_length", type=int, default=64,
                        help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for training")
    parser.add_argument("--frozen_layers", type=int, default=0,
                        help="Number of frozen layers in the model")
    parser.add_argument("--min_delta", type=float, default=0.00,
                        help="Minimum change in the monitored quantity to qualify as an improvement")
    parser.add_argument("--patience", type=int, default=2,
                        help="Number of epochs with no improvement after which training will be stopped")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = load_args()
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if args.save_model_dir is None:
        save_model_dir = Path().home() / f"Downloads/saved_model_{current_time}/"
    else:
        save_model_dir = Path(args.save_model_dir)

    # hf_model_name = "microsoft/deberta-base"
    # tokenizer_name = "microsoft/deberta-base"

    train_and_save_model(save_model_dir, tokenizer_name = args.tokenizer_name, hf_model_name = args.hf_model_name, max_epochs=args.max_epochs, batch_size=args.batch_size, max_length=args.max_length, learning_rate=args.learning_rate, frozen_layers=args.frozen_layers, min_delta=args.min_delta, patience=args.patience)