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
        self.bpe_encoder = TextEncoder(config.text.vocab_size)
        self.phoneme_encoder = TextEncoder(len(config.text.phonemes))
        self.text_encoder = TextEncoder()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, 
        *,
        audio, 
        audio_lengths,
        alignment
    ):

        # Check shapes
        # assert audio.shape[0] == len(alignment), "audio and alignment must have the same batch size"
        B = len(alignment)

        # Tokenize alignment text
        word_collections = []
        bpe_input = []
        bpe_length = []
        phonemes_input = []
        phonemes_length = []
        for i in range(B):
            current = alignment[i] # List of words, spaces and phonemes within

            # Tokenize each segment
            tokens = []
            words = []
            phonemes = []
            offset = 0
            phoneme_offset = 0
            spec_offset = 0
            for j in range(len(current)):
                word, duration, real_world = current[j][:3]
                encoded = self.tokenizer.encode(real_world)

                # Append to tokens list
                tokens += encoded

                # Append to words list
                if word is not None:
                    words.append(((offset, offset + len(encoded)), (spec_offset, spec_offset + duration), (phoneme_offset, (phoneme_offset + 1) if word is None else phoneme_offset + len(current[j][3]))))
                    
                # Append phonemes
                if word is not None:
                    for (phoneme, phoneme_duration) in current[j][3]:
                        phonemes.append((self.tokenizer.encode_phoneme(phoneme), phoneme_offset))
                        phoneme_offset += 1
                        spec_offset += phoneme_duration
                else:
                    phonemes.append((self.tokenizer.encode_phoneme("<SIL>"), phoneme_offset))
                    phoneme_offset += 1
                    spec_offset += duration

                # Update offset
                offset += len(encoded)

            # Append to batch
            word_collections.append(words)
            bpe_input.append(tokens)
            bpe_length.append(len(tokens))
            phonemes_input.append(phonemes)
            phonemes_length.append(len(phonemes))

        # Pad text inputs
        bpe_max_length = max(bpe_length)
        bpe_input_padded = torch.zeros(B, bpe_max_length, dtype=torch.long, device = audio.device)
        for i in range(B):
            bpe_input_padded[i, :bpe_length[i]] = torch.tensor(bpe_input[i])
        phoneme_max_length = max([len(x) for x in phonemes_input])
        phonemes_input_padded = torch.zeros(B, phoneme_max_length, dtype=torch.long, device = audio.device)
        for i in range(B):
            phonemes_input_padded[i, :len(phonemes_input[i])] = torch.tensor([x[0] for x in phonemes_input[i]])

        # Create masks
        bpe_mask = create_padding_mask(torch.tensor(bpe_length).to(bpe_input_padded.device), bpe_max_length, device = bpe_input_padded.device).unsqueeze(1)
        phonemes_mask = create_padding_mask(torch.tensor(phonemes_length).to(phonemes_input_padded.device), phoneme_max_length, device = phonemes_input_padded.device).unsqueeze(1)

        # Run BPE Encoder
        bpe_outputs = self.bpe_encoder(bpe_input_padded, bpe_mask)

        # Run Phoneme Encoder
        phoneme_outputs = self.phoneme_encoder(phonemes_input_padded, phonemes_mask)

        # Merge BPE and Phoneme outputs
        for i in range(B):
            for ((start_bpe, end_bpe), (start_spec, end_spec), (start_phone, end_phone)) in word_collections[i]:

                # Average BPE outputs per segment
                bpe_mean = torch.mean(bpe_outputs[i, start_bpe:end_bpe], dim=0)

                # Merge
                phoneme_outputs[i][start_phone:end_phone] += bpe_mean

        # Run Text Encoder
        token_outputs = self.text_encoder(phoneme_outputs, phonemes_mask)

        # Flatten embeddings filtering out silences
        text_embeddings_pre = []
        ind = 0
        for al in alignment:
            for segment in al:
                word, duration, src = segment[:3]
                offset = 0
                if word is not None:
                    text_embeddings_pre.append(token_outputs[ind, offset:offset + len(segment[3])])
                    offset += len(segment[3])
                else:
                    offset += 1
            ind += 1
        text_embeddings = torch.cat(text_embeddings_pre, dim=0)

        # Audio Embeddings
        audio_mask = create_padding_mask(audio_lengths, audio_lengths.max(), device = audio.device).unsqueeze(1)
        audio_outputs = self.audio_encoder(audio, audio_mask)
        audio_embeddings = audio_outputs.mean(dim=1)

        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)
        assert text_embeddings.shape[0] == audio_embeddings.shape[0], "text and audio embeddings must have the same batch size"

        # Compute loss
        labels = torch.arange(text_embeddings.shape[0]).to(audio.device)
        logit_scale = self.logit_scale.exp()
        logits_per_audio = logit_scale * audio_embeddings @ text_embeddings.T
        logits_per_text = logits_per_audio.T
        loss = (F.cross_entropy(logits_per_audio, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        # Return embeddings
        return text_embeddings, audio_embeddings, loss

        