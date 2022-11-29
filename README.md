# ttstokenizer: Tokenizer for Text to Speech (TTS) models

[![Version](https://img.shields.io/github/release/neuml/ttstokenizer.svg?style=flat&color=success)](https://github.com/neuml/ttstokenizer/releases)

`ttstokenizer` makes it easy to feed text to speech models with minimal dependencies that are Apache 2.0 compatible.

> See the original repository https://github.com/Kyubyong/g2p for more information on English Grapheme to Phoneme conversion.
>
> Other than removing unused dependencies and reorganizing the files, the original logic remains unchanged

The standard preprocessing logic for many English Text to Speech (TTS) models is as follows:

- Apply [Tacotron](https://github.com/keithito/tacotron) text normalization rules
  - This project replicates the logic found in [ESPnet](https://github.com/espnet/espnet_tts_frontend)
- Convert Graphemes to Phonemes
- Build an integer array mapping Phonemes to their integer token positions

This project adds a new tokenizer that runs the logic above. The output is consumable by machine learning models.

# Installation

The easiest way to install is via pip and PyPI

```
pip install ttstokenizer
```

## Usage

An example of tokenizing text for TTS models is shown below.

```
from ttstokenizer import TTSTokenizer

tokenizer = TTSTokenizer(tokens)
print(tokenizer("Text to tokenize"))

>>> array([ 4, 15, 10,  6,  4,  4, 28,  4, 34, 10,  2,  3, 51, 11])
```
