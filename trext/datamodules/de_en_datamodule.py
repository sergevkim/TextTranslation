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
        de_train_corpus_path = self.data_dir / 'train.de-en.de'
        en_train_corpus_path = self.data_dir / 'train.de-en.en'
        de_val_corpus_path = self.data_dir / 'val.de-en.de'
        en_val_corpus_path = self.data_dir / 'val.de-en.en'
        de_test_corpus_path = self.data_dir / 'test1.de-en.de'

        self.de_vocabulary, de_max_length = Vocabulary.build_vocabulary(
            text_corpus_paths=[
                de_train_corpus_path,
                de_val_corpus_path,
                de_test_corpus_path,
            ],
        )
        self.en_vocabulary, en_max_length = Vocabulary.build_vocabulary(
            text_corpus_paths=[
                en_train_corpus_path,
                en_val_corpus_path,
            ],
        )

        de_train_tags_lists = Editor.get_tags_lists(
            text_corpus_paths=[de_train_corpus_path],
            vocabulary=self.de_vocabulary,
            max_length=de_max_length,
        )
        en_train_tags_lists = Editor.get_tags_lists(
            text_corpus_paths=[en_train_corpus_path],
            vocabulary=self.en_vocabulary,
            max_length=en_max_length,
        )
        de_val_tags_lists = Editor.get_tags_lists(
            text_corpus_paths=[de_val_corpus_path],
            vocabulary=self.de_vocabulary,
            max_length=de_max_length,
        )
        en_val_tags_lists = Editor.get_tags_lists(
            text_corpus_paths=[en_val_corpus_path],
            vocabulary=self.en_vocabulary,
            max_length=en_max_length,
        )
        de_test_tags_lists = Editor.get_tags_lists(
            text_corpus_paths=[de_test_corpus_path],
            vocabulary=self.de_vocabulary,
            max_length=de_max_length,
        )

        train_data = dict(
            de_tags_lists=de_train_tags_lists,
            en_tags_lists=en_train_tags_lists,
        )
        val_data = dict(
            de_tags_lists=de_val_tags_lists,
            en_tags_lists=en_val_tags_lists,
        )
        test_data = dict(
            de_tags_lists=de_test_tags_lists,
            en_tags_lists=de_test_tags_lists, #TODO remove
        )

        return train_data, val_data, test_data

    def setup(self):
        train_data, val_data, test_data = self.prepare_data()

        self.train_dataset = DeEnDataset(
            de_tags_lists=train_data['de_tags_lists'][:1000],
            en_tags_lists=train_data['en_tags_lists'][:1000],
        )
        self.val_dataset = DeEnDataset(
            de_tags_lists=val_data['de_tags_lists'],
            en_tags_lists=val_data['en_tags_lists'],
        )

        self.test_dataset = DeEnDataset(
            de_tags_lists=test_data['de_tags_lists'],
            en_tags_lists=test_data['en_tags_lists'],
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
        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
        )

        return test_dataloader

