from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from ..utils import Editor, Vocabulary


class DeEnDataset(Dataset):
    def __init__(
            self,
            de_tags_lists: List[str],
            en_tags_lists: List[str],
        ):
        self.de_tags_lists = de_tags_lists
        self.en_tags_lists = en_tags_lists

    def __len__(self) -> int:
        return len(self.de_tags_lists)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        de_tags = torch.tensor(self.de_tags_lists[idx], dtype=torch.int64)
        en_tags = torch.tensor(self.en_tags_lists[idx], dtype=torch.int64)

        return (de_tags, en_tags)


class DeEnDataModule:
    def __init__(
            self,
            data_dir: Path,
            batch_size: int,
            num_workers: int,
        ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        de_corpus_path = self.data_dir / 'train.de-en.de'
        en_corpus_path = self.data_dir / 'train.de-en.en'

        self.de_vocabulary, de_max_length = Vocabulary.build_vocabulary(
            text_corpus_path=de_corpus_path,
        )
        self.en_vocabulary, en_max_length = Vocabulary.build_vocabulary(
            text_corpus_path=en_corpus_path,
        )

        de_tags_lists = Editor.get_tags_lists(
            text_corpus_path=de_corpus_path,
            vocabulary=self.de_vocabulary,
            max_length=de_max_length,
        )
        en_tags_lists = Editor.get_tags_lists(
            text_corpus_path=en_corpus_path,
            vocabulary=self.en_vocabulary,
            max_length=en_max_length,
        )

        data = dict(
            de_tags_lists=de_tags_lists,
            en_tags_lists=en_tags_lists,
        )

        return data

    def setup(self, val_ratio: float):
        data = self.prepare_data()
        full_dataset = DeEnDataset(
            de_tags_lists=data['de_tags_lists'],
            en_tags_lists=data['en_tags_lists'],
        )

        full_size = len(full_dataset)
        val_size = int(val_ratio * full_size)
        train_size = full_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size],
        )

    def train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return val_dataloader

    def test_dataloader(self):
        pass

