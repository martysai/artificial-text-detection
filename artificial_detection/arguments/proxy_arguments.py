import argparse

import torch


def set_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    general = parser.add_argument_group("General")
    general.add_argument("--df_path", default=None, type=str, help="Path to the translations dataframe.")
    general.add_argument("--metrics_names", nargs="*", type=str, default=[], help="Metrics names list.")
    general.add_argument("--output_path", default=None, type=str, help="Path to the output csv file.")

    comet_specific = parser.add_argument_group("Comet Specific")
    comet_specific.add_argument("--model_path", default=None, type=str, help="Path to the model.")

    # TODO: BLEURT specific

    return parser


def form_proxy_args():
    arg_parser = argparse.ArgumentParser(
        "Artificial Text Detection Proxy Metrics Module", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    set_args(arg_parser)
    known_args, _ = arg_parser.parse_known_args()
    known_args.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    return known_args
