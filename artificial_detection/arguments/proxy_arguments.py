import argparse


def set_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    checkpoints = parser.add_argument_group("Checkpoints")
    checkpoints.add_argument("--bin_ext", default="bin")
    return parser


def form_proxy_args():
    arg_parser = argparse.ArgumentParser(
        "Artificial Text Detection", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    set_args(arg_parser)
    known_args, _ = arg_parser.parse_known_args()
    known_args.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    return known_args
