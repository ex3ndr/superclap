import torch
import random
import textgrid
from pathlib import Path
from superclap.config import config
from superclap.audio import load_mono_audio, spectogram
from superclap.alignment import align_textgrid_with_source_text, extract_phonemes_in_words
from superclap.tokenizer import Tokenizer

def load_item(id):

    # Text
    with open(id + ".txt", 'r') as file:
        text = file.read()

    # TextGrid
    tg = textgrid.TextGrid.fromFile(id + ".TextGrid")

    # Audio
    waveform = load_mono_audio(id + ".flac", config.audio.sample_rate)
    spec = spectogram(waveform, config.audio.n_fft, config.audio.n_mels, config.audio.hop_size, config.audio.win_size, config.audio.mel_norm, config.audio.mel_scale, config.audio.sample_rate)
    spec = spec.T

    # Alignments
    al = align_textgrid_with_source_text(config, tg, text, spec.shape[0], id)
    if al is None:
        return None
    word_alignments, phoneme_alignments, combined_alignments = al

    # Split audio to phoneme segments
    audio_segments = []
    for (phoneme, start, end) in extract_phonemes_in_words(combined_alignments):
        if end - start > 80: # Ignore too long phonemes
            return None
        audio_segments.append(spec[start:end,:])

    # Results
    return waveform, spec, audio_segments, combined_alignments

def create_dataset_sampler(datasets):

    # Load valid files
    files = []
    for dataset in datasets:
        with open("./external_datasets/" + dataset + "/files_valid.txt", 'r') as file:
            lines = file.readlines()
        dataset_files = [("./external_datasets/" + dataset + "/" + l.strip()) for l in lines]
        files += dataset_files

    # Do sample
    def sample():
        while True:
            id = random.choice(files)
            it = load_item(id)
            if it is not None:
                _, _, audio_segments, alignments = it
                return audio_segments, alignments
            # else:
            #     print("Invalid item", id)

    return sample

def collate(batch):
    specs, alignments = zip(*batch)

    # Calculate spec lengths
    specs_lengths = []
    for spec in specs:
        for s in spec:
            specs_lengths.append(s.shape[0])

    # Pad specs
    max_len = max(specs_lengths)
    padded_specs = []
    for spec in specs:
        for s in spec:
            pad_size = max_len - s.shape[0]
            padded_spec = torch.nn.functional.pad(s, (0, 0, 0, pad_size))
            padded_specs.append(padded_spec)
    specs = padded_specs
    specs = torch.stack(specs)

    return specs, torch.tensor(specs_lengths), alignments

def load_dataset_loader(datasets, batch_size, num_workers):

    # Load sampler
    sampler = create_dataset_sampler(datasets)

    # Load dataset
    class WrappedDataset(torch.utils.data.IterableDataset):
        def __init__(self, sampler):
            self.sampler = sampler
        def generate(self):
            while True:
                yield self.sampler()
        def __iter__(self):
            return iter(self.generate())
    dataset = WrappedDataset(sampler)

    # Load loader
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, pin_memory = True, shuffle=False, collate_fn=collate)

    return loader

#
# Prepared Dataset
#

def load_prepared_item(path):
    state = torch.load(path, map_location="cpu")
    return state["spec"], state["phonemes"], state["tokens"], state["phonemes_duration"], state["len_tokens"], state["len_phonemes"]

def extract_training_data(src, phoneme_id = None):

    # Load record
    spec, phonemes, tokens, phonemes_duration, len_tokens, len_phonemes = src

    # Find phoneme indexes which match to phoneme_id
    phoneme_indexes = [i for i, phoneme in enumerate(phonemes) if phoneme == phoneme_id]

    # Pick random phoneme
    phoneme_index = random.choice(phoneme_indexes)

    # Extract spec part of phoneme
    start = phonemes_duration[:phoneme_index].sum().int().item()
    end = start + phonemes_duration[phoneme_index].int().item()
    spec = spec[start:end,:]

    # Find word index
    word_index = 0
    total_length = 0
    for length in len_phonemes:
        total_length += length
        if total_length <= phoneme_index:
            word_index += 1
        else:
            break
            
    # Find word first and last token
    word_start = len_tokens[:word_index].sum().int().item()
    word_end = word_start + len_tokens[word_index].int().item()

    return spec, tokens, phonemes, phoneme_index, word_start, word_end, len_tokens, len_phonemes


def load_prepared_sampler(src = None):
    suffix = ""
    if src is not None:
        suffix = "_" + src

    # Load all indexes
    datasets = {}
    for id in range(len(config.text.phonemes)):
        if Path("datasets/prepared" + suffix +"/phoneme_" + str(id) + ".txt").exists():
            with open("datasets/prepared" + suffix + "/phoneme_" + str(id) + ".txt", 'r') as file:
                lines = file.readlines()
                datasets[id] = [l.strip() for l in lines]

    # Sampler implementation
    tokenizer = Tokenizer()
    def sample(batch_size, phoneme_id = None, output_ids = False, output_phoneme_names = False):
        data = None

        # Parse phoneme id
        if type(phoneme_id) is str:
            phoneme_id = tokenizer.encode_phoneme(phoneme_id)

        # Load batch
        out_specs = []
        out_specs_length = []
        out_tokens = []
        out_tokens_length = []
        out_tokens_segment = []
        out_word_token_lengths = []
        out_word_phoneme_lengths = []
        out_phonemes = []
        out_phonemes_length = []
        out_phonemes_index = []
        out_ids = []
        out_phoneme_names = []
        for i in range(batch_size):

            # Resolve phoneme id
            ph_id = None
            if phoneme_id is None:
                ph_id = random.choice(list(datasets.keys()))
            else:
                ph_id = phoneme_id
            if output_phoneme_names:
                out_phoneme_names.append(config.text.phonemes[ph_id])

            # Pick random record
            id = random.choice(datasets[ph_id])
            if output_ids:
                out_ids.append(id)

            # Load record
            src = load_prepared_item("datasets/prepared" + suffix + "/" + id + ".pt")

            # Extract training data
            spec, tokens, phonemes, phoneme_index, word_start, word_end, word_token_lengths, word_phoneme_lengths = extract_training_data(src, phoneme_id = ph_id)

            # Append
            out_specs.append(spec)
            out_specs_length.append(spec.shape[0])
            out_tokens.append(tokens)
            out_tokens_length.append(len(tokens))
            out_phonemes.append(phonemes)
            out_phonemes_length.append(len(phonemes))
            out_phonemes_index.append(phoneme_index)
            out_tokens_segment.append([word_start, word_end])
            out_word_token_lengths.append(word_token_lengths)
            out_word_phoneme_lengths.append(word_phoneme_lengths)

        # Tensorize
        out_phonemes_index = torch.tensor(out_phonemes_index, dtype=torch.long)
        out_tokens_segment = torch.tensor(out_tokens_segment, dtype=torch.long)

        # Pad specs
        max_len = max(out_specs_length)
        padded_specs = []
        for spec in out_specs:
            pad_size = max_len - spec.shape[0]
            padded_spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_size))
            padded_specs.append(padded_spec)
        padded_specs = torch.stack(padded_specs)
        out_specs_length = torch.tensor(out_specs_length)
        
        # Pad tokens
        max_len = max(out_tokens_length)
        padded_tokens = []
        for tokens in out_tokens:
            pad_size = max_len - len(tokens)
            padded_tokens.append(torch.nn.functional.pad(tokens, (0, pad_size)))
        padded_tokens = torch.stack(padded_tokens)
        out_tokens_length = torch.tensor(out_tokens_length)

        # Pad phonemes
        max_len = max(out_phonemes_length)
        padded_phonemes = []
        for phonemes in out_phonemes:
            pad_size = max_len - len(phonemes)
            padded_phonemes.append(torch.nn.functional.pad(phonemes, (0, pad_size)))
        padded_phonemes = torch.stack(padded_phonemes)
        out_phonemes_length = torch.tensor(out_phonemes_length)

        output = (padded_specs, out_specs_length, padded_tokens, out_tokens_length, padded_phonemes, out_phonemes_length, out_phonemes_index, out_word_token_lengths, out_word_phoneme_lengths)
        if output_ids:
            output = output + (out_ids,)
        if output_phoneme_names:
            output = output + (out_phoneme_names,)
        return output

    return sample
            
