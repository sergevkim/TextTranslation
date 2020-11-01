from pathlib import Path
from typing import List, Tuple

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from trext.utils import Vocabulary


class DeEnDataset(Dataset):
    def __init__(
            self,
            de_sentences: List[str],
            en_sentences: List[str],
        ):
        self.de_sentences = de_sentences
        self.en_sentences = en_sentences

    def __len__(self) -> int:
        return len(self.de_sentences)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        de_sentence = Tensor(self.de_sentences[idx])
        en_sentence = Tensor(self.en_sentences[idx])

        return (de_sentence, en_sentence)


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
        de_filename = self.data_dir / 'train.de-en.de'
        en_filename = self.data_dir / 'train.de-en.en'

        de_vocabulary = Vocabulary.build_vocabulary(
            text_corpus_filename=de_filename,
        )
        en_vocabulary = Vocabulary.build_vocabulary(
            text_corpus_filename=en_filename,
        )


        data = dict(
            de_sentences=de_sentences,
            en_sentences=en_sentences,
        )

        return data

    def setup(self):
        data = self.prepare_data()
        full_dataset = DeEnDataset(
            de_sentences=data['de_sentences'],
            en_sentences=data['en_sentences'],
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

    def val_dataloder(self) -> DataLoader:
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return val_dataloader

    def test_dataloader(self):
        pass

