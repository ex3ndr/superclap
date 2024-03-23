import torch
import torchaudio
from .config import config
from .transformer import Transformer, ConvPositionEmbed

class AudioEncoder(torch.nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        # Transformer input
        self.transformer_input = torch.nn.Linear(config.audio.n_mels, config.audio_encoder.n_dim)

        # Convolutional positional encoder
        self.conv_embed = ConvPositionEmbed(n_dim = config.audio_encoder.n_dim, kernel_size = 31)

        # Transformer
        self.transformer = Transformer(

            # Architecture
            n_heads = config.audio_encoder.n_heads,
            n_layers = config.audio_encoder.n_layers,
            n_dim = config.audio_encoder.n_dim,
            n_dim_head = config.audio_encoder.n_dim_head,
            n_dim_ffn = config.audio_encoder.n_dim_ffn,

            # Dropout
            att_dropout = 0,
            ffn_dropout = 0.1,

            # Positional embedding
            position_embedding = 'alibi'
        )

    def forward(self, audio, mask):
        
        # Prepare
        audio = self.transformer_input(audio)
        audio = self.conv_embed(audio) + audio

        # Transformer
        audio = self.transformer(audio, mask = mask)

        return audio
