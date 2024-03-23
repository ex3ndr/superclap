import sentencepiece as spm
import os
from .config import config

class Tokenizer:
    def __init__(self):
        super(Tokenizer, self).__init__()
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(os.path.dirname(__file__), "..", "tokenizer.model"))
        self.phoneme_to_id = {token: i for i, token in enumerate(config.text.phonemes)}

    def encode(self, text):
        return self.sp.encode_as_ids(text)
    
    def encode_sample(self, text):
        return self.sp.encode(text, enable_sampling=True, alpha=0.1, nbest_size=-1)

    def encode_phoneme(self, token):
        return self.phoneme_to_id[token]
