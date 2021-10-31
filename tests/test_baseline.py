import os
import shutil
import torch

from unittest import TestCase

from detection import run
from detection.data.arguments import form_args


class TestBaseline(TestCase):
    def test_run(self):
        args = form_args()
        args.run_name = 'test_run'
        args.epochs = 1
        args.size = 128
        args.is_mock_data = True
        args.cuda = torch.cuda.is_available()
        args.device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.
                                   cuda.is_available() else 'cpu')
        run(args)
