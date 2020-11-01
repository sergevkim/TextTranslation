from pathlib import Path
from typing import List


class Editor:

    @staticmethod
    def get_lines(
            text_corpus_filename: Path,
        ):
        with open(text_corpus_filename) as f:
            lines = f.readlines()

        return lines

    @staticmethod
    def lines2tokens_lists(
            lines: List[str],
        ) -> List[List[str]]:
        tokens_lists = list()

        for line in lines:
            tokens = line.split()
            tokens_lists.append(tokens)

        return tokens_lists

    @staticmethod
    def tokens_lists2tags_lists(
            tokens_lists: List[List[str]],
            vocabulary,
            max_length,
        ) -> List[List[int]]:
        tags_lists = list()

        for tokens in tokens_lists:
            tags = list()

            for token in tokens:
                tag = vocabulary.token2tag[token]
                tags.append(tag)

            for i in range(1 + max_length - len(tags)):
                tags.append(vocabulary.token2tag['EOS'])

            tags_lists.append(tags)

        return tags_lists

    @classmethod
    def get_tags_lists(
            cls,
            text_corpus_filename: Path,
            vocabulary,
            max_length,
        ) -> List[List[int]]:
        lines = cls.get_lines(text_corpus_filename=text_corpus_filename)
        tokens_lists = cls.lines2tokens_lists(lines=lines)
        tags_lists = cls.tokens_lists2tags_lists(
            tokens_lists=tokens_lists,
            vocabulary=vocabulary,
            max_length=max_length,
        )

        return tags_lists

#FILE -> LINES -> TOKENS_LISTS -> TAGS_LISTS
#        LINE  -> TOKENS       -> TAGS

