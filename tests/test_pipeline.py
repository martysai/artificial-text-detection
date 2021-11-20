import torch

from unittest import TestCase

from detection.arguments import form_args
from detection.pipeline import pipeline
from tests import skip_github


class TestPipeline(TestCase):
    @skip_github
    def test_run(self):
        args = form_args()
        args.run_name = 'test_run'
        args.epochs = 1
        args.size = 8
        args.is_mock_data = True
        args.cuda = torch.cuda.is_available()
        args.device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.
                                   cuda.is_available() else 'cpu')
        pipeline(args)
