import pandas as pd
import unicodedata
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import os

MAXIUM_DATA_LENGTH = 10000

class myTokenizer:
    def __init__(self, num_words=8000, oov_token="<UNKNOWN>"):
        self.tokenizer = Tokenizer(
            num_words=num_words, 
            oov_token=oov_token,
            # filters='\"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n')
            filters='@#$%^&()*~\t\n'
        )
        self.num_words = num_words
        self.oov_token = oov_token

    def train_from_parquet(self, parquet_path, inputCol, outputCol):
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"not find this parquest file: {parquet_path}")
        df = pd.read_parquet(parquet_path).dropna().head(MAXIUM_DATA_LENGTH)
        df[inputCol] = df[inputCol].apply(self.clean_text)
        df[outputCol] = df[outputCol].apply(self.clean_text)
        texts = df[inputCol].tolist() + df[outputCol].tolist()
        self.tokenizer.fit_on_texts(texts)
        print(f"✅ Tokenizer is build, word size: {len(self.tokenizer.word_index) + 1}")

    def train_from_json(self, json_path, inputCol, outputCol):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"not find this parquest file: {json_path}")
        df = pd.read_json(json_path, lines=True).dropna().head(MAXIUM_DATA_LENGTH)
        df[inputCol] = df[inputCol].apply(self.clean_text)
        df[outputCol] = df[outputCol].apply(self.clean_text)
        texts = df[inputCol].tolist() + df[outputCol].tolist()
        self.tokenizer.fit_on_texts(texts)
        print(f"✅ Tokenizer is build, word size: {len(self.tokenizer.word_index) + 1}")

    def save_tokenizer(self, save_path="myTokenizer.pkl"):

        save_dir = os.path.join(os.getcwd(), 'tokenizer')
        os.makedirs(save_dir, exist_ok=True)
        acutual_save_path = os.path.join(save_dir, save_path)
        os.makedirs(os.path.dirname(acutual_save_path), exist_ok=True)
        with open(acutual_save_path, "wb") as f:
            pickle.dump(self.tokenizer, f)
        print(f"✅ Tokenizer is saved at: {acutual_save_path}")

    @staticmethod
    def load_tokenizer(load_path="myTokenizer.pkl"):
        with open(load_path, "rb") as f:
            tokenizer = pickle.load(f)
        print(f"✅ Tokenizer is loaded successfully: {load_path}")
        return tokenizer
    
    @staticmethod
    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    
    @staticmethod
    def split_numbers_and_units(text):
        text = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', text)  # 25kg -> 25 kg
        text = re.sub(r'(\$)(\d+)', r'\1 \2', text)          # $415 -> $ 415
        return text
    
    @staticmethod
    def clean_text(text):
        text = myTokenizer.unicode_to_ascii(text.lower().strip())
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"([?.!,])", r" \1 ", text) 
        text = re.sub(r"\s+", " ", text).strip()

        text =  "<start> " +  text + " <end>"
        return text

    def get_tokenizer(self):
        return self.tokenizer
