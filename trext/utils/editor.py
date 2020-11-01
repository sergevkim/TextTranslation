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
            tokens = lines.split()
            tokens_lists.append(tokens)

        return tokens_lists

    @staticmethod
    def tokens_lists2tags_lists(
            tokens_lists: List[List[str]],
            vocabulary,
        ) -> List[List[int]]:
        tags_lists = list()

        for tokens in tokens_lists:
            tags = list()

            for token in tokens:
                tag = vocabulary.token2tag[token]
                tags.append(tag)

            tags_lists.append(tags)

        return tags_lists

#FILE -> LINES -> TOKENS_LISTS -> TAGS_LISTS
#        LINE  -> TOKENS       -> TAGS

