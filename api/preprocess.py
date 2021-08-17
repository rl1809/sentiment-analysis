import json
import re
import string

import emoji
from pyvi import ViTokenizer

RAW_FILEPATH = '../data/raw/'
PREPROCESS_FILE_PATH = '../data/preprocess/'


with open('replace.json') as json_file:
    replace = json.load(json_file)

REPLACE_LIST = replace['replace_list']


EMOJIS = ''.join(emoji.UNICODE_EMOJI['en'].keys())
PATTERN = string.punctuation+string.digits+EMOJIS
TABLE = str.maketrans(PATTERN, " "*len(PATTERN))


def normalize_text(text):
    """Normalize text data"""
    text = ' '+text+' '
    text = text.lower()
    for key, value in REPLACE_LIST.items():
        text = text.replace(key, value)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'www\S+', ' ', text)

    text = text.translate(TABLE)

    text = re.sub(r'\s+', ' ', text).strip()

    text = ViTokenizer.tokenize(text)

    return text