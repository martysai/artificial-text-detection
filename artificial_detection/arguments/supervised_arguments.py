import argparse
import torch


def set_supervised_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    train_args = parser.add_argument_group("Training arguments")
    train_args.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_args.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_args.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    train_args.add_argument("--model_name", type=str, default="T5", help="Model name")

    return parser


def form_supervised_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser("ATD: Supervised Arguments")
    set_supervised_args(arg_parser)
    known_args, _ = arg_parser.parse_known_args()
    known_args.device = torch.device(
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )
    return known_args
