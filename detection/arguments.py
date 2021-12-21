from typing import List, Optional

import argparse
import os
import os.path as path

import torch


def get_dataset_path(dataset_name: str, langs: Optional[List[str]] = None, ext: str = "bin") -> str:
    dir_path = path.dirname(path.dirname(path.realpath(__file__)))
    dvc_path = path.join(dir_path, "resources/data")
    if langs:
        dataset_real_name = f"{dataset_name}.{langs[0]}-{langs[1]}.{ext}"
    else:
        dataset_real_name = f"{dataset_name}.{ext}"
    return path.join(dvc_path, dataset_real_name)


def set_args(parser: argparse.ArgumentParser):
    checkpoints = parser.add_argument_group("Checkpoints")
    checkpoints.add_argument("--model_path", type=str, default="model.pth", help="Model checkpoint path")
    checkpoints.add_argument("--bin_ext", default="bin")
    checkpoints.add_argument("--ds_ext", default="pth")

    prefix = path.dirname(os.getcwd())

    train_args = parser.add_argument_group("Training arguments")
    train_args.add_argument("--easy_nmt_batch_size", type=int, default=16)
    train_args.add_argument("--easy_nmt_model_name", type=str, default="opus-mt")
    train_args.add_argument("--easy_nmt_offline", type=bool, default=False)
    # TODO-EasyNMT: improve for many models
    train_args.add_argument(
        "--offline_prefix",
        type=str,
        default=f"{prefix}/resources/data/opus-mt",
        help="Define the absolute path where the model is stored",
    )
    train_args.add_argument(
        "--offline_cache_prefix",
        type=str,
        default=f"{prefix}/resources/data/",
        help="Define he path to the directory where to store cache",
    )
    train_args.add_argument("--dataset_name", type=str, default="tatoeba", help="dataset name which will be loaded.")
    train_args.add_argument("--epochs", type=int, default=50, help="# epochs")
    train_args.add_argument("--train_batch", type=int, default=16, help="train batch size")
    train_args.add_argument("--eval_batch", type=int, default=16, help="eval batch size")
    train_args.add_argument("--log_steps", type=int, default=10, help="# steps for logging")
    train_args.add_argument("--size", type=int, default=10000, help="# samples in the mock/tatoeba/wikimatrix dataset")
    train_args.add_argument("--warmup_steps", type=int, default=100)
    train_args.add_argument("--weight_decay", type=int, default=1e-4)
    train_args.add_argument("--is_mock_data", type=bool, default=False)

    libraries = parser.add_argument_group("Libraries")
    libraries.add_argument("--run_name", type=str, default="default", help="A name of run to be used in wandb")
    return parser


def form_args():
    arg_parser = argparse.ArgumentParser(
        "Artificial Text Detection", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    set_args(arg_parser)
    known_args, _ = arg_parser.parse_known_args()
    known_args.cuda = torch.cuda.is_available()
    known_args.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    return known_args
