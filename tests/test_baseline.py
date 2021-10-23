import os
import shutil
import torch

from unittest import TestCase

from detection import run
from detection.data.arguments import form_args


class TestBaseline(TestCase):
    def test_run(self):
        if os.environ.get('GITHUB_ACTIONS'):
            pass
        args = form_args()
        args.run_name = 'test_run'
        args.epochs = 1
        args.size = 128
        args.is_mock_data = True
        args.cuda = torch.cuda.is_available()
        args.device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.
                                   cuda.is_available() else 'cpu')
        trainer = run(args)
        test_model_name = 'test_model.pth'
        trainer.model.save_pretrained(test_model_name)

        if os.path.exists(test_model_name):
            shutil.rmtree(test_model_name)
