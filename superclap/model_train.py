import torch
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

    def forward(self, 
        *,
        audio, 
        alignment
    ):

        # Check shapes
        assert audio.shape[0] == len(alignment), "audio and alignment must have the same batch size"
        B = audio.shape[0]

        # Tokenize alignment text
        word_collections = []
        spec_durations = []
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
            spec_durations.append(spec_offset)
            bpe_input.append(tokens)
            bpe_length.append(len(tokens))
            phonemes_input.append(phonemes)
            phonemes_length.append(len(phonemes))

        # Pad text inputs
        bpe_max_length = max(bpe_length)
        bpe_input_padded = torch.zeros(B, bpe_max_length, dtype=torch.long)
        for i in range(B):
            bpe_input_padded[i, :bpe_length[i]] = torch.tensor(bpe_input[i])
        phoneme_max_length = max([len(x) for x in phonemes_input])
        phonemes_input_padded = torch.zeros(B, phoneme_max_length, dtype=torch.long)
        for i in range(B):
            phonemes_input_padded[i, :len(phonemes_input[i])] = torch.tensor([x[0] for x in phonemes_input[i]])

        # Create masks
        bpe_mask = create_padding_mask(torch.tensor(bpe_length).to(bpe_input_padded.device), bpe_max_length, device = bpe_input_padded.device).unsqueeze(1)
        phonemes_mask = create_padding_mask(torch.tensor(phonemes_length).to(phonemes_input_padded.device), phoneme_max_length, device = phonemes_input_padded.device).unsqueeze(1)

        # Run BPE Encoder
        bpe_outputs = self.bpe_encoder(bpe_input_padded, bpe_mask)

        # Run Phoneme Encoder
        phoneme_outputs = self.phoneme_encoder(phonemes_input_padded, phonemes_mask)

        # Combine outputs and unwrap to word per batch
        text_pre = []
        for i in range(B):
            for ((start_bpe, end_bpe), (start_spec, end_spec), (start_phone, end_phone)) in word_collections[i]:

                # Average BPE outputs per segment
                bpe_mean = torch.mean(bpe_outputs[i, start_bpe:end_bpe], dim=0)

                # Phoneme outputs in word
                phonemes_values = phoneme_outputs[i][start_phone:end_phone]

                # Combine values
                combined_values = phonemes_values + bpe_mean

                # Append to text_pre
                text_pre.append(combined_values)

        # Pad text_pre to max length
        NB = len(text_pre)
        text_pre_max_length = max([x.shape[0] for x in text_pre])
        text_pre_padded = torch.zeros(NB, text_pre_max_length, text_pre[0].shape[1])
        for i in range(NB):
            text_pre_padded[i, :text_pre[i].shape[0]] = text_pre[i]

        # Create mask
        text_pre_mask = create_padding_mask(torch.tensor([x.shape[0] for x in text_pre]).to(text_pre_padded.device), text_pre_max_length, device = text_pre_padded.device).unsqueeze(1)

        # Text outputs
        text_outputs = self.text_encoder(text_pre_padded, text_pre_mask)

        # Text embeddings
        text_embeddings = torch.mean(text_outputs, dim=1)

        # Create audio mask
        audio_mask = create_padding_mask(torch.tensor(spec_durations).to(audio.device), audio.shape[1], device = audio.device).unsqueeze(1)

        # Run audio encoder
        audio_pre_outputs = self.audio_encoder(audio, audio_mask)

        # Extract audio segments
        audio_embeddings = []
        for i in range(B):
            for ((start_bpe, end_bpe), (start_spec, end_spec), (start_phone, end_phone)) in word_collections[i]:
                audio_segment = audio_pre_outputs[i, start_spec:end_spec].mean(0)
                audio_embeddings.append(audio_segment)
        audio_embeddings = torch.stack(audio_embeddings)

        # Return embeddings
        return text_embeddings, audio_embeddings

        