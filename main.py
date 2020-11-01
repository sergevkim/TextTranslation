from argparse import ArgumentParser
from pathlib import Path

import torch

from trext.datamodules import DeEnDataModule
from trext.loggers import NeptuneLogger
#from trext.models import SimpleTranslator
from trext.trainer import Trainer
from trext.utils import Editor, Vocabulary


def main(args):
    #model = SimpleTranslator()
    datamodule = DeEnDataModule(
        data_dir = Path('data/homework_machine_translation_de-en'),
        batch_size=2,
        num_workers=4,
    )
    datamodule.setup(val_ratio=0.1)
    loader = datamodule.train_dataloader()

    print(len(loader))
    for i, b in enumerate(loader):
        print(b[0])
        break
    '''
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

