from pathlib import Path
from typing import List, Tuple

from torchtext.data import BucketIterator, Field
from torchtext.datasets import TranslationDataset


class DeEnBucketsDataModule:
    def __init__(
            self,
            data_path: Path,
            batch_size: int,
            num_workers: int,
        ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.SRC = Field(tokenize=lambda x: x.split(),
            tokenizer_language="de",
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
        )
        self.TRG = Field(tokenize=lambda x: x.split(),
            tokenizer_language="en",
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
        )

    def setup(self):
        self.prepare_data()
        self.train_dataset = TranslationDataset(
            str(self.data_path / 'train.de-en.'),
            ['de', 'en'],
            fields=(self.SRC, self.TRG),
        )
        self.val_dataset = TranslationDataset(
            str(self.data_path / 'val.de-en.'),
            ['de', 'en'],
            fields=(self.SRC, self.TRG),
        )
        self.test_dataset = TranslationDataset(
            str(self.data_path / 'test1.de-en.'),
            ['de', 'de'],
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

    def train_dataloader(self):
        train_iterator = BucketIterator(
            self.train_dataset,
            batch_size=self.batch_size,
            sort_key=lambda x: len(x.comment_text),
        )

        return train_iterator

    def val_dataloader(self):
        val_iterator = BucketIterator(
            self.val_dataset,
            batch_size=self.batch_size,
            sort_key=lambda x: len(x.comment_text),
        )

        return val_iterator

    def test_dataloader(self):
        test_iterator = BucketIterator(
            self.test_dataset,
            batch_size=self.batch_size,
            sort_key=lambda x: len(x.comment_text),
        )

        return test_iterator
