"""
Tokenizer module
"""

import os
import re

import numpy as np

from anyascii import anyascii

from .expand import normalize_numbers
from .g2p import G2p, dirname


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

        # Override vocab
        self.vocab = {
            **self.loadvocab(os.path.join(dirname, "cmudict.dict")),
            **self.loadvocab(os.path.join(dirname, "custom.dict"))
        }

        # Multi word terms (separated by hyphens)
        self.multiterm = {
            x.replace("-", " ").strip():x for x in
            sorted([x for x in self.vocab if "-" in x], key=lambda x: x.count("-"), reverse=True)
        }

        # Abbrevations (periods replaced with zzz)
        # Only support multiple period abbrevations)
        self.multiterm.update({
            x.replace("zzz", "."):x for x in
            sorted(
                [x for x in self.vocab if "zzz" in x[:-1]], key=lambda x: x.count("zzz"),
                reverse=True
            )
        })

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

    def punctuation(self):
        """
        Gets a list of punctuation token ids.

        Returns:
            list of punctuation token ids
        """

        return [v for k, v in self.tokens.items() if k in ".,!?;"]

    def lookup(self, word, pos):
        # Primary stress for nouns, adjectives, conjunctions and personal pronouns
        primary = pos.startswith(("N", "J", "CC", "PRP"))
        return self.vocab[word][0] if primary else self.vocab[word][-1]

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
        text = self.dates(text)
        text = text.lower()
        text = normalize_numbers(text)
        text = self.expand(text)
        text = self.symbols(text)

        # Combine multi-term entries into a single term
        for k, v in self.multiterm.items():
            text = re.sub(fr"\b{re.escape(k)}\b", v, text)

        # Make uppercase to work with parent stripping accented chars logic
        text = text.upper()

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)

        return text

    def dates(self, text):
        """
        Normalizes dates.

        Args:
            text: input text

        Return:
            normalized text
        """

        text = re.sub(r"(\d+) BC", r"\1 b.c.", text)
        text = re.sub(r"(\d+) AD", r"\1 a.d.", text)

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

    def loadvocab(self, path):
        """
        Loads a vocabulary file from path. This file has a word to pronunciation mapping.
        Pronunciations use the ARPABET codes.

        Args:
            path: path to vocab file

        Returns:
            {word: [pronunciations]}
        """

        vocab = {}

        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    word, pron = line.strip().split(maxsplit=1)

                    # Normalize word
                    word = re.sub(r"\(\d+\)", r"", word).lower().strip()

                    # Create special entries for multi-period abbrevations
                    if "." in word and (not word.endswith(".") or word.count(".") > 1):
                        word = word.replace(".", "zzz")

                    # Remove trailing comments, and split into tokens
                    pron = re.sub(r"(.+?)#.*", r"\1", pron).strip().split()

                    # Initialize word
                    if word not in vocab:
                        vocab[word] = []

                    # Add pronunciation
                    vocab[word].append(pron)

        return vocab


class IPATokenizer(TTSTokenizer):
    """
    International Phonetic Alphabet (IPA) TTS Tokenizer - converts English Graphemes to
    IPA Tokens Ids
    """

    def __init__(self, tokenize=True, transcribe=True):
        """
        Creates a new IPA tokenizer instance.

        Args:
            tokenize: if True, IPA token ids are returned (default) otherwise, the IPA text is
                      returned
            transcribe: if True, input text is transcribed to IPA (default), otherwise it is assumed
                        that the text is already transcribed to IPA
        """

        # Call parent constructor. This tokenizer has it's own token mapping and requires spaces
        super().__init__(None, False)

        # Enables tokenization to IPA token ids
        self.tokenize = tokenize

        # Enables text to IPA transcription
        self.transcribe = transcribe

    def __call__(self, text):
        if self.transcribe:
            # Tokenize text using base tokenization method
            tokens = super().__call__(text)

            # Get ARPABET to IPA vocab mapping
            vocab = self.ipavocab()

            # Map ARPABET tokens to IPA tokens and join as a string
            text = "".join(vocab.get(x, x) for x in tokens)
        else:
            text = text.strip()

        # IPA token lookup
        tokens = self.ipatokens()

        # Lookup token ids and return
        return np.array([tokens.get(x, x) for x in text], dtype=np.int64) if self.tokenize else text

    def punctuation(self):
        """
        Gets a list of punctuation token ids.

        Returns:
            list of punctuation token ids
        """

        return [v for k, v in self.ipatokens().items() if k in ".,!?;"]

    def ipavocab(self):
        """
        Builds a vocabulary mapping the ARPABET alphabet (via CMUdict) to the
        International Phonetic Alphabet (IPA).

        See the following links for more information on this mapping.
          - https://github.com/Kyubyong/g2p/issues/29
          - https://en.wikipedia.org/wiki/ARPABET

        Returns:
            ARPABET to IPA vocabulary mapping
        """

        return {
            '<pad>': '<pad>',
            '<unk>': '<unk>',
            '<s>': '<s>',
            '</s>': '</s>',
            'AA0': 'ɑ',
            'AA1': 'ˈɑː',
            'AA2': 'ˌɑ',
            'AE0': 'æ',
            'AE1': 'ˈæ',
            'AE2': 'ˌæ',
            'AH0': 'ə',
            'AH1': 'ˈʌ',
            'AH2': 'ˌʌ',
            'AO0': 'ɔ',
            'AO1': 'ˈɔː',
            'AO2': 'ˌɔ',
            'AW0': 'aʊ',
            'AW1': 'ˈaʊ',
            'AW2': 'ˌaʊ',
            'AY0': 'aɪ',
            'AY1': 'ˈaɪ',
            'AY2': 'ˌaɪ',
            'B': 'b',
            'CH': 'tʃ',
            'D': 'd',
            'DH': 'ð',
            'EH0': 'ɛ',
            'EH1': 'ˈɛ',
            'EH2': 'ˌɛ',
            'ER0': 'ɚ',
            'ER1': 'ˈɚ',
            'ER2': 'ˌɚ',
            'EY0': 'eɪ',
            'EY1': 'ˈeɪ',
            'EY2': 'ˌeɪ',
            'F': 'f',
            'G': 'ɡ',
            'HH': 'h',
            'IH0': 'ɪ',
            'IH1': 'ˈɪ',
            'IH2': 'ˌɪ',
            'IY0': 'i',
            'IY1': 'ˈiː',
            'IY2': 'ˌi',
            'JH': 'dʒ',
            'K': 'k',
            'L': 'l',
            'M': 'm',
            'N': 'n',
            'NG': 'ŋ',
            'OW0': 'oʊ',
            'OW1': 'ˈoʊ',
            'OW2': 'ˌoʊ',
            'OY0': 'ɔɪ',
            'OY1': 'ˈɔɪ',
            'OY2': 'ˌɔɪ',
            'P': 'p',
            'R': 'ɹ',
            'S': 's',
            'SH': 'ʃ',
            'T': 't',
            'TH': 'θ',
            'UH0': 'ʊ',
            'UH1': 'ˈʊ',
            'UH2': 'ˌʊ',
            'UW0': 'u',
            'UW1': 'ˈuː',
            'UW2': 'ˌu',
            'V': 'v',
            'W': 'w',
            'Y': 'j',
            'Z': 'z',
            'ZH': 'ʒ',
        }

    def ipatokens(self):
        """
        Builds an International Phonetic Alphabet (IPA) token to token id mapping.

        This grapheme mapping is from here: https://hf.co/hexgrad/Kokoro-82M/blob/main/kokoro.py#L74

        Returns:
            {token: token id}
        """

        # pylint: disable=C0301
        pad = "$"
        punctuation = ';:,.!?¡¿—…"«»“” '
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        ipaletters = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        symbols = [pad] + list(punctuation) + list(letters) + list(ipaletters)
        mapping = {}
        for x, token in enumerate(symbols):
            mapping[token] = x

        return mapping
