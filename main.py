from argparse import ArgumentParser
from pathlib import Path

import torch

from trext.datamodules import DeEnDataModule
from trext.loggers import NeptuneLogger
#from trext.models import SimpleTranslator
from trext.trainer import Trainer
from trext.utils import Editor, Vocabulary


def main(args):
    en_vocabulary, en_max_length = Vocabulary.build_vocabulary(
        text_corpus_filename=Path('./data/homework_machine_translation_de-en/train.de-en.en')
    )
    de_vocabulary, de_max_length = Vocabulary.build_vocabulary(
        text_corpus_filename=Path('./data/homework_machine_translation_de-en/train.de-en.de')
    )

    a = Editor.get_tags_lists(
        text_corpus_filename=Path('./data/homework_machine_translation_de-en/train.de-en.de'),
        vocabulary=de_vocabulary,
        max_length = de_max_length,
    )
    print(a[0], len(a[0]))

    '''
    model = SimpleTranslator()
    datamodule = DeEnDataModule()
    datamodule.setup()
    logger = NeptuneLogger()
    trainer = Trainer()

    trainer.fit(
        model=model,
        datamodule=datamodule,
        logger=logger,
    )

    trainer.predict(
        model=model,
        datamodule=datamodule,
    )
    '''

if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    args = dict(
        attn_dim=8,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        enc_emb_dim=256,
    )
    main(args)

