from pathlib import Path
from typing import List, Tuple

import torch
from torchtext.data import BucketIterator, Field, Iterator
from torchtext.datasets import TranslationDataset


class DeEnBucketsDataModule:
    def __init__(
            self,
            data_path: Path,
            batch_size: int,
            num_workers: int,
            device: torch.device,
        ):
        self.device = device

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        self.SRC = Field(
            tokenize=lambda x: x.split(),
            tokenizer_language="de",
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
        )
        self.TRG = Field(
            tokenize=lambda x: x.split(),
            tokenizer_language="en",
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
        )

    def setup(self) -> None:
        self.prepare_data()
        self.train_dataset = TranslationDataset(
            path=str(self.data_path / 'train.de-en.'),
            exts=['de', 'en'],
            fields=(self.SRC, self.TRG),
        )
        self.val_dataset = TranslationDataset(
            path=str(self.data_path / 'val.de-en.'),
            exts=['de', 'en'],
            fields=(self.SRC, self.TRG),
        )
        self.test_dataset = TranslationDataset(
            path=str(self.data_path / 'test1.de-en.'),
            exts=['de', 'de'],
            fields=(self.SRC, self.SRC),
        )

        self.SRC.build_vocab(
            self.train_dataset,
            min_freq=2,
        )
        self.TRG.build_vocab(
            self.train_dataset,
            min_freq=2,
        )

    def train_dataloader(self) -> Iterator:
        train_iterator = BucketIterator(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            device=self.device,
        )

        return train_iterator

    def val_dataloader(self) -> Iterator:
        val_iterator = BucketIterator(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            device=self.device,
        )

        return val_iterator

    def test_dataloader(self) -> Iterator:
        test_iterator = BucketIterator(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            device=self.device,
        )

        return test_iterator

