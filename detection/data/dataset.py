import pickle
import zlib
from typing import Any, Optional

import torch


class TextDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    @staticmethod
    def load(dataset_path: str, suffix: Optional[str] = 'train') -> Any:
        with open(f'{dataset_path}.{suffix}', 'rb') as file:
            compressed_dataset = file.read()
            dumped_dataset = zlib.decompress(compressed_dataset)
            dataset = pickle.loads(dumped_dataset)
        return dataset

    def save(self, dataset_path: str, suffix: Optional[str] = 'train') -> None:
        with open(f'{dataset_path}.{suffix}', 'wb') as file:
            dumped_dataset = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
            compressed_dataset = zlib.compress(dumped_dataset)
            file.write(compressed_dataset)

    def to(self, device: Optional[str] = None):
        if not device:
            return self
        self.encodings = self.encodings.to(device)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
