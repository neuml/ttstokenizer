# ttstokenizer: Tokenizer for Text to Speech (TTS) models

[![Version](https://img.shields.io/github/release/neuml/ttstokenizer.svg?style=flat&color=success)](https://github.com/neuml/ttstokenizer/releases)

> See the original repository https://github.com/Kyubyong/g2p for more information on English Grapheme to Phoneme conversion.
>
> Other than removing unused dependencies and reorganizing the files, the original logic remains intact.

`ttstokenizer` makes it easy to feed text to speech models with minimal dependencies that are Apache 2.0 compatible.

The standard preprocessing logic for many English Text to Speech (TTS) models is as follows:

- Apply [Tacotron](https://github.com/keithito/tacotron) text normalization rules
  - This project replicates the logic found in [ESPnet](https://github.com/espnet/espnet_tts_frontend)
- Convert Graphemes to Phonemes
- Build an integer array mapping Phonemes to their integer token positions

This project adds new tokenizers that runs the logic above. The output is consumable by machine learning models.

# Installation

The easiest way to install is via pip and PyPI

```
pip install ttstokenizer
```

## Usage

This project has two supported tokenizers.

- `TTSTokenizer` - Tokenizes text to [ARPABET](https://en.wikipedia.org/wiki/ARPABET) phoenemes. Word to phoeneme definitions are provided by [CMUdict](https://github.com/cmusphinx/cmudict). These phonemes are then mapped to token ids using a provided token - token id mapping.

- `IPATokenizer` - Tokenizes text to [International Phonetic Alphabet (IPA)](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) phoenemes. The
graphemes for each phoneme are mapped to token ids.

The `IPATokenizer` is designed to be a drop in replacement for models that depend on eSpeak to tokenize text into IPA phoenemes.

An example of tokenizing text for each of the TTS models is shown below.

```python
from ttstokenizer import TTSTokenizer

tokenizer = TTSTokenizer(tokens)
print(tokenizer("Text to tokenize"))

>>> array([ 4, 15, 10,  6,  4,  4, 28,  4, 34, 10,  2,  3, 51, 11])
```

```python
from ttstokenizer import IPATokenizer

tokenizer = IPATokenizer()
print(tokenizer("Text to tokenize"))

>>> array([ 62 156  86  53  61  62  16  62  70  16  62 156  57 135  53  70  56 157 43 102  68])
```

## Debugging

Both tokenizers also support returning raw types to help with debugging.

The following returns ARPABET phonemes instead of token ids for the `TTSTokenizer`.

```python
from ttstokenizer import TTSTokenizer

tokenizer = TTSTokenizer()
print(tokenizer("Text to tokenize"))

>>> ['T', 'EH1', 'K', 'S', 'T', 'T', 'AH0', 'T', 'OW1', 'K', 'AH0', 'N', 'AY2', 'Z']
```

The same can be done with the IPATokenizer. The following returns the transcribed IPA tokens

```python
from ttstokenizer import IPATokenizer

tokenizer = IPATokenizer(tokenize=False)
print(tokenizer("Text to tokenize"))

>>> "tˈɛkst tɐ tˈoʊkɐnˌaɪz"
```

The `IPATokenizer` has an additional method to accept IPA tokens directly.

```python
from ttstokenizer import IPATokenizer

tokenizer = IPATokenizer(transcribe=False)
print(tokenizer("tˈɛkst tɐ tˈoʊkɐnˌaɪz"))

>>> array([[ 62 156  86  53  61  62  16  62  70  16  62 156  57 135  53  70  56 157 43 102  68]])
```

Notice how the output is the same as above. When the output doesn't sound right, these methods can help trace what's going on.
