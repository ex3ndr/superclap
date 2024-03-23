import torch
from .config import Config
from .model_audio import AudioEncoder
from .model_text import TextEncoder
from .tokenizer import Tokenizer

class SuperCLAP(torch.nn.Module):
     def __init__(self):
        super(SuperVoice, self).__init__()
        self.tokenizer = Tokenizer()
        self.audio_encoder = AudioEncoder()
        self.text_encoder = TextEncoder()

    def forward(self, 
        *,

        # Mel Spectogram
        input_audio, 

        # Phonemes + Text + Indexes of used phonemes
        input_text,
        input_phonemes
    ):
        
        # Check shapes
        assert isinstance(input_audio, list), "input_audio must be a list"
        assert len(input_audio) == input_phonemes.shape[0], "input_audio and input_phonemes must have the same batch size"
        assert input_text.shape[0] == input_phonemes.shape[0], "input_text and input_phonemes must have the same batch size"
        assert input_indexes.shape[0] == input_phonemes.shape[0], "input_indexes and input_phonemes must have the same batch size"

        # audio_embedding = self.audio_encoder(input_audio, input_audio_length)
        # text_embedding = self.text_encoder(input_text, input_phonemes, input_indexes)
        # return audio_embedding, text_embedding