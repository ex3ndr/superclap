import torch
import torchaudio
from .config import config

class AudioEncoder(torch.nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

    def forward(self, audio, audio_length):
        raise NotImplementedError
