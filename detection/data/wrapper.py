import copy
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as torch_data
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2


class TextDetectionDataset(torch_data.Dataset):
    def __init__(self, encodings, labels, device: Optional[str] = None):
        self.encodings = encodings
        self.labels = labels
        self.device = device if device else 'cpu'

    @staticmethod
    def get_corpus(targets: List[str], translations: List[str]):
        corpus = [''] * (2 * len(targets))
        corpus[::2] = translations
        corpus[1::2] = targets
        return corpus

    @staticmethod
    def load(dataset_path: str, device: Optional[str] = None) -> torch_data.Dataset:
        dataset_settings = torch.load(dataset_path)
        if device:
            dataset_settings['device'] = device
        return TextDetectionDataset(**dataset_settings)

    @staticmethod
    def load_csv(csv_path: str, device: Optional[str] = 'cpu') -> torch_data.Dataset:
        df = pd.read_csv(csv_path)
        corpus = TextDetectionDataset.get_corpus(df['targets'].values.tolist(),
                                                 df['translations'].values.tolist())
        labels = torch.FloatTensor([0, 1] * (len(corpus) // 2))
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        encodings = tokenizer(corpus, truncation=True, padding=True)
        encodings, labels = TextDetectionDataset.to_device(encodings, labels, device=device)
        dataset = TextDetectionDataset(encodings, labels, device=device)
        return dataset

    @staticmethod
    def to_device(encodings: torch.FloatTensor, labels: torch.FloatTensor, device: Optional[str] = 'cpu'):
        if device:
            # TODO-Extra: написать на GPU получше
            try:
                encodings = encodings.to(device)
                labels = labels.to(device)
            except AttributeError:
                pass
        return encodings, labels

    @staticmethod
    def get_encodings_from_range(encodings: Any, objects_range: List[int]) -> Any:
        new_encodings = copy.deepcopy(encodings)
        for key, val in encodings.items():
            new_encodings[key] = np.array(val)[objects_range].tolist()
        return new_encodings

    def split(self) -> Tuple[torch_data.Dataset, torch_data.Dataset]:
        # TODO-Extra: написать на GPU получше
        if hasattr(self.encodings, 'detach'):
            encodings = self.encodings.detach().cpu().numpy()
            labels = self.labels.detach().cpu().numpy()
        else:
            encodings = self.encodings
            labels = self.labels

        objects_range = list(range(len(labels)))

        train_range, eval_range, train_labels, eval_labels = train_test_split(
            objects_range,
            labels,
            test_size=TEST_SIZE
        )
        train_encodings = TextDetectionDataset.get_encodings_from_range(encodings, train_range)
        eval_encodings = TextDetectionDataset.get_encodings_from_range(encodings, eval_range)
        train_labels, eval_labels = torch.FloatTensor(train_labels), torch.FloatTensor(eval_labels)

        train_dataset = TextDetectionDataset(train_encodings, train_labels, self.device)
        eval_dataset = TextDetectionDataset(eval_encodings, eval_labels, self.device)
        return train_dataset, eval_dataset

    def save(self, dataset_path: str) -> None:
        torch.save({
            'encodings': self.encodings,
            'labels': self.labels,
            'device': self.device
        }, dataset_path)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)
