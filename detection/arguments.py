import argparse
import os
import os.path as path

import torch


def set_args(parser: argparse.ArgumentParser):
    checkpoints = parser.add_argument_group("Checkpoints")
    checkpoints.add_argument("--model_path", type=str, default="model.pth", help="Model checkpoint path")
    checkpoints.add_argument("--save_bin", action="store_true", help="if passed, saving .bin")
    checkpoints.add_argument("--bin_ext", default="bin")
    checkpoints.add_argument("--ds_ext", default="pth")

    prefix = path.dirname(os.getcwd())

    train_args = parser.add_argument_group("Training arguments")
    train_args.add_argument("--easy_nmt_batch_size", type=int, default=16)
    train_args.add_argument("--easy_nmt_model_name", type=str, default="opus-mt")
    train_args.add_argument("--easy_nmt_offline", type=bool, default=False)
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
    train_args.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
    train_args.add_argument("--train_batch", type=int, default=16, help="train batch size")
    train_args.add_argument("--eval_batch", type=int, default=16, help="eval batch size")
    train_args.add_argument("--log_steps", type=int, default=10, help="# steps for logging")
    train_args.add_argument("--size", type=int, default=10000, help="# samples in the mock/tatoeba/wikimatrix dataset")
    train_args.add_argument("--warmup_steps", type=int, default=100)
    train_args.add_argument("--weight_decay", type=int, default=0.0)
    train_args.add_argument("--is_mock_data", type=bool, default=False)

    experiments = parser.add_argument_group("Experiments")
    experiments.add_argument("--detector_dataset_path", type=str, default="", help="dataset where labeling is stored")
    experiments.add_argument(
        "--detector_dataset_test_path", type=str, default="", help="dataset where test dataset is stored"
    )
    experiments.add_argument("--run_name", type=str, default="default", help="A name of run to be used in wandb")
    experiments.add_argument("--target_name", type=str, default="target")
    experiments.add_argument("--unsupervised_target_name", type=str, default="unsupervised_target")
    return parser


def form_args():
    arg_parser = argparse.ArgumentParser(
        "Artificial Text Detection", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    set_args(arg_parser)
    known_args, _ = arg_parser.parse_known_args()
    known_args.cuda = torch.cuda.is_available()
    known_args.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    # known_args.device = "cpu"

    if known_args.easy_nmt_offline:
        prefix_name = known_args.offline_prefix[known_args.offline_prefix.rfind("/") + 1 :]
        if prefix_name.startswith("opus"):
            known_args.model_name = "opus-mt"
        elif prefix_name.startswith("m2m100"):
            known_args.model_name = "m2m100"
        elif prefix_name.startswith("mbart"):
            known_args.model_name = "mbart"
        else:
            raise AttributeError("Wrong EasyNMT offlne prefix name")
    else:
        known_args.model_name = known_args.easy_nmt_model_name

    return known_args
