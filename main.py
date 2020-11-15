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

from config import ConfigDataClass


def set_seed(seed=9):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


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
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    tags = [trg_field.vocab.stoi[trg_field.init_token]]
    unk_indices = list()

    for i in range(max_len):
        trg_tensor = torch.LongTensor(tags).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:,-1].item()

        if pred_token == trg_field.vocab.stoi[trg_field.unk_token]:
            unk_indices.append(i)

        tags.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    result_tokens = list()
    tags = tags[1:]

    for i, tag in enumerate(tags):
        if i not in unk_indices:
            result_tokens.append(trg_field.vocab.itos[tag])
        else:
            result_tokens.append(tokens[i])

    #trg_tokens = [trg_field.vocab.itos[tag] for tag in tags]
    #return trg_tokens[1:], attention

    return result_tokens, attention


def main(args):
    set_seed(seed=9)
    start_time = time.time()
    print(f"Device: {args['device']}")

    datamodule = DeEnBucketsDataModule(
        data_path=args['data_path'],
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        device=args['device'],
    )
    datamodule.setup()
    datamodule_time = time.time()
    print(f"Datamodule is prepared ({datamodule_time - start_time:.2f} seconds)")

    args['src_pad_idx'] = datamodule.SRC.vocab.stoi[datamodule.SRC.pad_token]
    args['trg_pad_idx'] = datamodule.TRG.vocab.stoi[datamodule.TRG.pad_token]
    args['input_dim'] = len(datamodule.SRC.vocab)
    args['output_dim'] = len(datamodule.TRG.vocab)

    translator = TransformerTranslator(
        src_pad_idx=args['src_pad_idx'],
        trg_pad_idx=args['trg_pad_idx'],
        learning_rate=args['learning_rate'],
        input_dim=args['input_dim'],
        output_dim=args['output_dim'],
        hidden_dim=args['hidden_dim'],
        encoder_dropout_p=args['encoder_dropout_p'],
        encoder_heads_num=args['encoder_heads_num'],
        encoder_layers_num=args['encoder_layers_num'],
        encoder_pf_dim=args['encoder_dff_dim'],
        decoder_dropout_p=args['decoder_dropout_p'],
        decoder_heads_num=args['decoder_heads_num'],
        decoder_layers_num=args['decoder_layers_num'],
        decoder_pf_dim=args['decoder_dff_dim'],
        device=args['device'],
    ).to(args['device'])

    if not args['inference_only']:
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

        checkpoint = torch.load(f'models/v{args["version"]}-e{args["max_epoch"]}.hdf5', map_location=args['device'])

    else:
        checkpoint = torch.load(f'models/v{args["version"]}-e15.hdf5', map_location=args['device'])

    translator.load_state_dict(checkpoint['model_state_dict'])

    f = open('test1.de-en.en', 'w')

    for i in range(2998):
        src = vars(datamodule.test_dataset.examples[i])['src']

        translation, attention = translate_sentence(src, datamodule.SRC, datamodule.TRG, translator, args['device'])
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
        decoder_dropout_p=0.1,  #TODO
        decoder_heads_num=8,  #TODO
        decoder_hidden_dim=512,  #TODO
        decoder_layers_num=6,  #TODO
        decoder_dff_dim=2048,  #TODO
        device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
        encoder_dropout_p=0.1,  #TODO
        encoder_heads_num=8,  #TODO
        encoder_hidden_dim=512,  #TODO
        encoder_layers_num=6,  #TODO
        encoder_dff_dim=2048,  #TODO
        hidden_dim=512,
        learning_rate=2e-4,
        max_epoch=40,
        num_workers=4,
        verbose=True,
        version='1.5',
        inference_only=True,
    )

    main(args)

