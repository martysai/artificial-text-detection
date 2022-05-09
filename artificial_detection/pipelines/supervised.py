import argparse

from artificial_detection.models.detectors import SimpleDetector
from artificial_detection.arguments import form_supervised_args


class SupervisedPipeline:
    pass


def main(args: argparse.Namespace) -> None:
    pipeline = SupervisedPipeline()
    pipeline.init(args)
    trained_model = pipeline.train()
    collected_results = pipeline.evaluate(trained_model)
    pipeline.save(trained_model, collected_results)


if __name__ == "__main__":
    supervised_args = form_supervised_args()
    main(supervised_args)
