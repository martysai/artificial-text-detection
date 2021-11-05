from typing import Any, Optional

import torch


class TextDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, device: Optional[str] = None):
        self.encodings = encodings
        self.labels = labels
        self.device = device if device else 'cpu'

    @staticmethod
    def load(dataset_path: str, suffix: Optional[str] = 'train') -> Any:
        dataset_settings = torch.load(f'{dataset_path}.{suffix}')
        return TextDetectionDataset(**dataset_settings)

    def save(self, dataset_path: str, suffix: Optional[str] = 'train') -> None:
        torch.save({
            'encodings': self.encodings,
            'labels': self.labels,
            'device': self.device
        }, f'{dataset_path}.{suffix}')

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
