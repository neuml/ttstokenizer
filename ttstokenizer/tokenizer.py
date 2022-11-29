"""
Tokenizer module
"""

import re

import numpy as np

from anyascii import anyascii

from .expand import normalize_numbers
from .g2p import G2p


class TTSTokenizer(G2p):
    """
    Text to Speech (TTS) Tokenizer - converts English Graphemes to Phoneme Tokens Ids
    """

    def __init__(self, tokens=None, nospace=True):
        """
        Creates a new tokenizer instance. Optionally can pass a list of phoneme tokens to use to map
        outputs to token id arrays.

        Args:
            tokens: list of phoneme tokens - uses order to infer phoneme token ids
            nospace: if space phoneme tokens should be removed, defaults to True
        """

        # Call parent constructor
        super().__init__()

        # Build map of phoneme token to id
        self.tokens = {token:x for x, token in enumerate(tokens)} if tokens else {}

        # Remove phoneme space tokens
        self.nospace = nospace

        # List of expansions
        self.expansions = [(re.compile(fr"\b{x[0]}\.", re.IGNORECASE), x[1]) for x in [
            ('mrs', 'misess'),
            ('mr', 'mister'),
            ('dr', 'doctor'),
            ('st', 'saint'),
            ('co', 'company'),
            ('jr', 'junior'),
            ('maj', 'major'),
            ('gen', 'general'),
            ('drs', 'doctors'),
            ('rev', 'reverend'),
            ('lt', 'lieutenant'),
            ('hon', 'honorable'),
            ('sgt', 'sergeant'),
            ('capt', 'captain'),
            ('esq', 'esquire'),
            ('ltd', 'limited'),
            ('col', 'colonel'),
            ('ft', 'fort'),
        ]]

    def __call__(self, text):
        # Normalize text
        text = self.normalize(text)

        # Convert to phonemes
        tokens = super().__call__(text)

        # Remove whitespace tokens
        if self.nospace:
            tokens = [x for x in tokens if x != " "]

        # Build phoneme token id array and return
        return np.array([self.tokens[x] for x in tokens], dtype=np.int64) if self.tokens else tokens

    def normalize(self, text):
        """
        Applies text normalization and cleaning routines for English text.

        Args:
            text: input text

        Returns:
            normalized text
        """

        # Clean and normalize text
        text = anyascii(text)
        text = text.lower()
        text = normalize_numbers(text)
        text = self.expand(text)
        text = self.symbols(text)
        text = text.upper()
        text = re.sub(r"\s+", " ", text)

        return text

    def expand(self, text):
        """
        Runs a set of text expansions.

        Args:
            text: input text

        Returns:
            expanded text
        """

        for regex, replacement in self.expansions:
            text = re.sub(regex, replacement, text)

        return text

    def symbols(self, text):
        """
        Expands and cleans symbols from text.

        Args:
            text: input text

        Returns:
            clean text
        """

        # Expand symbols
        text = re.sub(r"\;", ",", text)
        text = re.sub(r"\:", ",", text)
        text = re.sub(r"\-", " ", text)
        text = re.sub(r"\&", "and", text)

        # Clean unnecessary symbols
        return re.sub(r'[\(\)\[\]\<\>\"]+', '', text)
