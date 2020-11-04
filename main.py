import time
from argparse import ArgumentParser
from pathlib import Path

import torch

from trext.datamodules import DeEnDataModule
#from trext.loggers import NeptuneLogger
from trext.models import AttentionTranslator, Encoder, Decoder, Attention
from trext.trainer import Trainer
from trext.utils import Editor, Vocabulary


def main(args):
    start_time = time.time()
    print(f"Device is: {args['device']}")

    print("Preparing datamodule...")
    datamodule = DeEnDataModule(
        data_dir=Path('data/homework_machine_translation_de-en'),
        batch_size=args['batch_size'],
        num_workers=4,
    )
    datamodule.setup()
    print(f"Datamodule is prepared ({time.time() - start_time} seconds)")

    encoder = Encoder(
        input_dim=len(datamodule.de_vocabulary),
        embedding_dim=args['encoder_embedding_dim'],
        encoder_hidden_dim=args['encoder_hidden_dim'],
        decoder_hidden_dim=args['decoder_hidden_dim'],
        dropout_p=args['encoder_dropout_p'],
    )
    attention = Attention(
        encoder_hidden_dim=args['encoder_hidden_dim'],
        decoder_hidden_dim=args['decoder_hidden_dim'],
    )
    decoder = Decoder(
        output_dim=len(datamodule.en_vocabulary),
        embedding_dim=args['decoder_embedding_dim'],
        encoder_hidden_dim=args['encoder_hidden_dim'],
        decoder_hidden_dim=args['decoder_hidden_dim'],
        dropout_p=args['decoder_dropout_p'],
        attention=attention,
    )
    translator = AttentionTranslator(
        encoder=encoder,
        decoder=decoder,
        teacher_forcing_ratio=0.5,
        learning_rate=3e-4,
        device=args['device'],
    ).to(args['device'])

    '''
    loader = datamodule.train_dataloader()
    print(len(loader))
    for i, b in enumerate(loader):
        print(b[0])
        break
    '''

    trainer = Trainer(
        logger=None,
        max_epoch=args['max_epoch'],
        verbose=args['verbose'],
        version=args['version'],
    )

    print('Let\'s start training!')
    trainer.fit(
        model=translator,
        datamodule=datamodule,
    )

    print('Predicts!')
    predicts = trainer.predict(
        model=translator,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    args = dict(
        batch_size=64,
        decoder_dropout_p=0.5,
        decoder_hidden_dim=128,
        decoder_embedding_dim=128,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        encoder_dropout_p=0.5,
        encoder_hidden_dim=128,
        encoder_embedding_dim=128,
        max_epoch=10,
        verbose=True,
        version='0.1',
    )
    main(args)

