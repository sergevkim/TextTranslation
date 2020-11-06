import random
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from trext.datamodules import DeEnBucketsDataModule
#from trext.loggers import NeptuneLogger
from trext.models import AttentionTranslator, Encoder, Decoder, Attention
from trext.trainer import Trainer
from trext.utils import Editor, Vocabulary


def set_seed(seed=9):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    

def main(args):
    set_seed()

    datamodule = DeEnBucketsDataModule(
        data_path=args['data_path'],
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
    )
        
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    encoder = TransformerEncoder(
        INPUT_DIM, 
        HID_DIM, 
        ENC_LAYERS, 
        ENC_HEADS, 
        ENC_PF_DIM, 
        args['encoder_dropout_p'], 
        device,
    ).to(device)

    decoder = TransformerDecoder(
        OUTPUT_DIM, 
        HID_DIM, 
        DEC_LAYERS, 
        DEC_HEADS, 
        DEC_PF_DIM, 
        args['decoder_dropout_p'],
        device,
    ).to(device)

    translator = TransformerTranslator(
        encoder=encoder,
        decoder=decoder,
        source_pad_idx=SRC_PAD_IDX,
        target_pad_idx=TRG_PAD_IDX,
        learning_rate=3e-4,
        device=device,
    ).to(device)

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
        decoder_dropout_p=0.1,
        decoder_hidden_dim=128,
        decoder_embedding_dim=128,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        encoder_dropout_p=0.1,
        encoder_hidden_dim=128,
        encoder_embedding_dim=128,
        max_epoch=10,
        verbose=True,
        version='0.1',
    )
    main(args)

