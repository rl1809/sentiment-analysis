"""Preprocess raw data and encode data for model"""
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer

PRETRAINED_TOKENIZER_PATH = 'vinai/phobert-base'


class BertTokenizer:
    """Transform text data to numberical to feed to model"""
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TOKENIZER_PATH)
        self.label_encoder = LabelEncoder()
    
    def fit(self, content, label):
        self.label_encoder.fit(label)

    def tokenize_text(self, text):
        encoded_text = self.tokenizer(
                text,
                add_special_tokens = True,
                max_length = 50,
                return_token_type_ids = False,
                pad_to_max_length='right',
                return_attention_mask = True,
                return_tensors = 'pt'
            )
        return encoded_text
    
    def encode_label(self, label):
        encoded_label = self.label_encoder.transform(label)
        return encoded_label

    def __call__(self, data, label):
        encoded_data = self.tokenize_text(data)
        encoded_label = self.encode_label([label])[0]
        return encoded_data, encoded_label

class TextTokenizer:
    def __init__(self, max_len=128):
        self.tokenizer = Tokenizer(filters='+')
        self.label_encoder = LabelEncoder()
        self.max_len = max_len
    
    def fit(self,data, label):
        self.tokenizer.fit_on_texts(data)
        self.word_index = self.tokenizer.word_index
        self.label_encoder.fit(label)

    def tokenize_text(self, text):
        encoded_text = self.tokenizer.texts_to_sequences([text]) 
        encoded_text = pad_sequences(encoded_text, maxlen=self.max_len,
                                     padding='post',truncating='post')
        return encoded_text[0]
    
    
    def encode_label(self, label):
        encoded_label = self.label_encoder.transform([label])[0]
        return encoded_label
    

    def __call__(self, data, label):
        encoded_data = self.tokenize_text(data)
        encoded_label = self.encode_label(label)
        return encoded_data, encoded_label

def create_embedding_matrix(word_index, embedding_dict):
    embedding_matrix = np.zeros((len(word_index)+1,100))
    for word, i in word_index.items():
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
    return embedding_matrix

