import torch

from trext.datamodules import DeEnDataModule
from trext.loggers import NeptuneLogger
from trext.models import SimpleTranslator
from trext.trainer import Trainer


def main(args):
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


if __name__ == "__main__":
    args = dict(
        attn_dim=8,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        enc_emb_dim=256,
    )
    main(args)

