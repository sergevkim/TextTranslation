import random
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from trext.datamodules import DeEnBucketsDataModule
#from trext.loggers import NeptuneLogger
from trext.models import AttentionTranslator, Encoder, Decoder, Attention
from trext.models import TransformerTranslator, TransformerEncoder, TransformerDecoder
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

    start_time = time.time()
    datamodule = DeEnBucketsDataModule(
        data_path=args['data_path'],
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
    )
    datamodule.setup()
    print(f"Datamodule is prepared ({time.time() - start_time} seconds)")

    args['src_pad_idx'] = datamodule.SRC.vocab.stoi[datamodule.SRC.pad_token]
    args['trg_pad_idx'] = datamodule.TRG.vocab.stoi[datamodule.TRG.pad_token]
    args['input_dim'] = len(datamodule.SRC.vocab)
    args['output_dim'] = len(datamodule.TRG.vocab)

    new_translator = TransformerTranslator(
        src_pad_idx=args['src_pad_idx'],
        trg_pad_idx=args['trg_pad_idx'],
        learning_rate=args['learning_rate'],
        input_dim=args['input_dim'],
        output_dim=args['output_dim'],
        hidden_dim=args['hidden_dim'],
        encoder_dropout_p=args['encoder_dropout_p'],
        encoder_heads_num=args['encoder_heads_num'],
        encoder_layers_num=args['encoder_layers_num'],
        decoder_dropout_p=args['decoder_dropout_p'],
        decoder_heads_num=args['decoder_heads_num'],
        decoder_layers_num=args['decoder_layers_num'],
    )
    '''
    SRC_PAD_IDX = datamodule.SRC.vocab.stoi[datamodule.SRC.pad_token]
    TRG_PAD_IDX = datamodule.TRG.vocab.stoi[datamodule.TRG.pad_token]
    INPUT_DIM = len(datamodule.SRC.vocab)
    OUTPUT_DIM = len(datamodule.TRG.vocab)

    encoder = TransformerEncoder(
        INPUT_DIM,
        HID_DIM,
        ENC_LAYERS,
        ENC_HEADS,
        ENC_PF_DIM,
        args['encoder_dropout_p'], 
        args['device'],
    ).to(args['device'])

    decoder = TransformerDecoder(
        OUTPUT_DIM,
        HID_DIM,
        DEC_LAYERS,
        DEC_HEADS,
        DEC_PF_DIM,
        args['decoder_dropout_p'],
        args['device'],
    ).to(args['device'])

    translator = TransformerTranslator(
        encoder=encoder,
        decoder=decoder,
        source_pad_idx=SRC_PAD_IDX,
        target_pad_idx=TRG_PAD_IDX,
        learning_rate=3e-4,
        device=args['device'],
    ).to(args['device'])
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

    checkpoint = torch.load(f'models/v{args["version"]}-e{args["max_epoch"] - 1}.hdf5', map_location=args['device'])

    model = TransformerTranslator(
        encoder=encoder,
        decoder=decoder,
        source_pad_idx=SRC_PAD_IDX,
        target_pad_idx=TRG_PAD_IDX,
        learning_rate=3e-4,
        device=args['device'],
    ).to(args['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args['device'])
    print(model.device)

    f = open('test1.de-en.en', 'w')

    def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
        model.eval()

        if isinstance(sentence, str):
            nlp = spacy.load('de')
            tokens = [token.text.lower() for token in nlp(sentence)]
        else:
            tokens = [token.lower() for token in sentence]

        tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        src_indexes = [src_field.vocab.stoi[token] for token in tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
        src_mask = model.make_source_mask(src_tensor)

        with torch.no_grad():
            enc_src = model.encoder(src_tensor, src_mask)

        trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_mask = model.make_target_mask(trg_tensor)

            with torch.no_grad():
                output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            pred_token = output.argmax(2)[:,-1].item()
            trg_indexes.append(pred_token)

            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                break

        trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

        return trg_tokens[1:], attention

    for i in range(2998):
        src = vars(datamodule.train_dataset.examples[i])['src']

        translation, attention = translate_sentence(src, datamodule.SRC, datamodule.TRG, model, args['device'])
        translation.pop()

        print(' '.join(translation), file=f)
        if i % 10 == 0:
            print(i, ' '.join(translation))


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    args = dict(
        batch_size=64,
        data_path=Path('homework_machine_translation_de-en'),
        decoder_dropout_p=0.1,
        decoder_heads_num=8,
        decoder_hidden_dim=128,
        decoder_layers_num=3,
        decoder_pf_dim=512,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        encoder_dropout_p=0.1,
        encoder_heads_num=8,
        encoder_hidden_dim=128,
        encoder_layers_num=3,
        encoder_pf_dim=512,
        hidden_dim=256,
        max_epoch=10,
        num_workers=4,
        verbose=True,
        version='1.0',
    )

    main(args)

