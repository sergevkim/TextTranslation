from pathlib import Path
from typing import List

from trext.utils.editor import Editor


class Vocabulary:
    def __init__(self):
        self.token2tag = dict()
        self.token2count = dict()
        self.tag2token = dict()
        self.tag2token[0] = 'SOS'
        self.tag2token[1] = 'EOS'
        self.n_tokens = 2
        self.max_length = 0

    def add_token(self, token: str):
        if token not in self.token2tag:
            self.token2tag[token] = self.n_tokens
            self.token2count[token] = 1
            self.tag2token[self.n_tokens] = token
            self.n_tokens += 1
        else:
            self.token2count[token] += 1

    @staticmethod
    def build_vocabulary(
            text_corpus_filename: Path,
        ):
        vocabulary = Vocabulary()

        lines = Editor.get_lines(text_corpus_filename=text_corpus_filename)

        for line in lines:
            tokens = line.split()
            self.max_length = max(len(tokens), self.max_length)

            for token in tokens:
                self.add_token(token)

        return vocabulary

