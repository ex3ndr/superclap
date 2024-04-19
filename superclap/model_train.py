import torch
import torch.nn.functional as F
import numpy as np
from .config import config
from .model_audio import AudioEncoder
from .model_text import TextEncoder
from .tokenizer import Tokenizer
from .masks import create_padding_mask

class SuperCLAPTrainer(torch.nn.Module):
    def __init__(self):
        super(SuperCLAPTrainer, self).__init__()
        self.tokenizer = Tokenizer()
        self.audio_encoder = AudioEncoder()
        self.token_encoder = TextEncoder(config.text.vocab_size)
        self.phoneme_encoder = TextEncoder(len(config.text.phonemes))
        self.meta_encoder = TextEncoder()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def audio_embeddings(self, audio, audio_lengths):
        audio_mask = create_padding_mask(audio_lengths, audio_lengths.max(), device = audio.device).unsqueeze(1)
        audio_outputs = self.audio_encoder(audio, audio_mask)
        audio_embeddings = audio_outputs.mean(dim=1)
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)
        return audio_embeddings

    def forward(self, 
        *,
        audio, 
        audio_length,
        tokens,
        tokens_length,
        phonemes,
        phonemes_length,
        word_token_lengths,
        word_phoneme_lengths,
        phonemes_index,
    ):
        B = len(audio)

        # Create masks
        audio_mask = create_padding_mask(audio_length, audio.shape[1], device = audio.device).unsqueeze(1)
        tokens_mask = create_padding_mask(tokens_length, tokens.shape[1], device = tokens.device).unsqueeze(1)
        phonemes_mask = create_padding_mask(phonemes_length, phonemes.shape[1], device = phonemes.device).unsqueeze(1)

        # Run token encoder
        token_outputs = self.token_encoder(tokens, tokens_mask)

        # Run phoneme encoder
        phoneme_outputs = self.phoneme_encoder(phonemes, phonemes_mask)

        # Merge token and phoneme embeddings to meta inputs
        meta_inputs = phoneme_outputs
        for i in range(B):
            phoneme_offset = 0
            token_offset = 0
            for (token_length, phoneme_length) in zip(word_token_lengths[i], word_phoneme_lengths[i]):

                # Often silence between words are coded as zero tokens
                if (token_length == 0):
                    phoneme_offset += phoneme_length
                    token_offset += token_length
                    continue

                # Get token and phoneme offsets
                token_start = token_offset
                token_end = token_offset + token_length
                phoneme_start = phoneme_offset
                phoneme_end = phoneme_offset + phoneme_length

                # Read token output and average it
                token_output = torch.mean(token_outputs[i, token_start:token_end], dim=0)

                # Throw exception if token_output contains NaN values
                if torch.isnan(token_output).any():
                    raise ValueError("token_output contains NaN values.")

                # Add to phoneme output
                meta_inputs[i][phoneme_start:phoneme_end] += token_output

                # Advance
                phoneme_offset += phoneme_length
                token_offset += token_length

        # Meta encoder
        meta_embeddings = self.meta_encoder(meta_inputs, phonemes_mask)

        # Audio encoder
        audio_outputs = self.audio_encoder(audio, audio_mask)

        # Compute embeddings
        meta_embeddings = meta_embeddings[torch.arange(meta_embeddings.shape[0]), phonemes_index]
        audio_embeddings = audio_outputs.mean(dim=1) # How to make it respect mask?

        # Normalize embeddings
        meta_embeddings = F.normalize(meta_embeddings, dim=-1)
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)
        assert meta_embeddings.shape[0] == audio_embeddings.shape[0], "meta and audio embeddings must have the same batch size"

        # Compute loss
        labels = torch.arange(meta_embeddings.shape[0]).to(audio.device)
        logit_scale = self.logit_scale.exp()
        logits_per_audio = logit_scale * audio_embeddings @ meta_embeddings.T
        logits_per_meta = logits_per_audio.T
        loss = (F.cross_entropy(logits_per_audio, labels) + F.cross_entropy(logits_per_meta, labels)) / 2

        # Return embeddings
        return meta_embeddings, audio_embeddings, loss

        