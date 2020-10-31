from pathlib import Path
from typing import List


class Vocabulary:
    def __init__(self):
        self.word2idx = dict()
        self.word2cnt = dict()
        self.idx2word = dict()
        self.idx2word[0] = 'SOS'
        self.idx2word[1] = 'EOS'
        self.n_words = 2

    def add_sentence(self, sentence: str):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word: str):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.word2cnt[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2cnt[word] += 1


def prepare_vocabulary(
        text_corpus_filename: Path,
    ) -> Vocabulary:
    vocabulary = Vocabulary()

    with open(text_corpus_filename) as f:
        lines = f.readlines()

    print(lines[0])
    for line in lines:
        vocabulary.add_sentence(
            sentence=line
        )

    return vocabulary

