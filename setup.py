# pylint: disable = C0111
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    DESCRIPTION = f.read()

setup(
    name = "ttstokenizer",
    version = "1.0.0",
    description = "Tokenizer for Text to Speech (TTS) models",
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    url = "https://github.com/neuml/ttstokenizer",
    license="Apache 2.0: http://www.apache.org/licenses/LICENSE-2.0",
    packages = ["ttstokenizer"],
    install_requires = [
        "anyascii>=0.3.1",
        "inflect>=0.3.1",
        "numpy>=1.13.1",
        "nltk>=3.2.4"
    ],
    include_package_data = True
)
