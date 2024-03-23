import torch
from .config import config
from .transformer import Transformer

class TextEncoder(torch.nn.Module):
    def __init__(self, in_features = None):
        super(TextEncoder, self).__init__()
        self.in_features = in_features
        if in_features is not None:
            self.embedding = torch.nn.Embedding(in_features, config.text_encoder.n_dim)
            torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.transformer = Transformer(

            # Architecture
            n_heads = config.text_encoder.n_heads,
            n_layers = config.text_encoder.n_layers,
            n_dim = config.text_encoder.n_dim,
            n_dim_head = config.text_encoder.n_dim_head,
            n_dim_ffn = config.text_encoder.n_dim_ffn,

            # Dropout
            att_dropout = 0,
            ffn_dropout = 0.1,

            # Positional embedding
            position_embedding = 'alibi'
        )

    def forward(self, x, mask = None):
        if self.in_features is not None:
            return self.transformer(self.embedding(x), mask = mask)
        else:
            return self.transformer(x, mask = mask)
