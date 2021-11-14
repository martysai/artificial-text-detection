import argparse
import os.path as path

import torch


def get_dataset_path(dataset_name: str, ext: str = 'bin') -> str:
    dir_path = path.dirname(path.dirname(path.realpath(__file__)))
    dvc_path = path.join(dir_path, "resources/data")
    return path.join(dvc_path, f"{dataset_name}.{ext}")


def set_args(parser: argparse.ArgumentParser):
    checkpoints = parser.add_argument_group('Checkpoints')
    checkpoints.add_argument('--model_path', type=str, default='model.pth',
                             help='Model checkpoint path')
    checkpoints.add_argument('--bin_ext', default='bin')
    checkpoints.add_argument('--ds_ext', default='pth')

    train_args = parser.add_argument_group('Training arguments')
    train_args.add_argument('--dataset_name', type=str, default='tatoeba',
                            help='dataset name which will be loaded.')
    train_args.add_argument('--epochs', type=int, default=50,
                            help='# epochs')
    train_args.add_argument('--train_batch', type=int, default=16,
                            help='train batch size')
    train_args.add_argument('--eval_batch', type=int, default=16,
                            help='eval batch size')
    train_args.add_argument('--log_steps', type=int, default=10,
                            help='# steps for logging')
    train_args.add_argument('--size', type=int, default=514000,
                            help='# samples in the tatoeba dataset')
    train_args.add_argument('--warmup_steps', type=int, default=100)
    train_args.add_argument('--weight_decay', type=int, default=1e-4)
    train_args.add_argument('--is_mock_data', type=bool, default=False)

    libraries = parser.add_argument_group('Libraries')
    libraries.add_argument('--run_name', type=str, default='default',
                           help='A name of run to be used in wandb')
    return parser


def form_args():
    arg_parser = argparse.ArgumentParser(
        'Artificial Text Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    set_args(arg_parser)
    known_args, _ = arg_parser.parse_known_args()
    known_args.cuda = torch.cuda.is_available()
    known_args.device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.
                                     cuda.is_available() else 'cpu')
    return known_args
