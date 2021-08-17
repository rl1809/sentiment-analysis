"""Clean raw text and save to preprocess folder"""

import os
import json
import re
import string

import emoji
import pandas as pd
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


def create_csv_file():
    """Save preprocess data as csv files"""
    file_names = os.listdir(RAW_FILEPATH)

    for name in file_names:
        file_path = os.path.join(RAW_FILEPATH, name)
        dataframe = pd.read_csv(file_path)
        dataframe['content'] = dataframe.apply(lambda x: normalize_text(x))
        save_path = os.path.join(PREPROCESS_FILE_PATH, name)
        dataframe.to_csv(save_path)


if __name__ == '__main__':
    create_csv_file()
