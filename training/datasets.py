import torch
import random
import textgrid
from superclap.config import config
from superclap.audio import load_mono_audio, spectogram
from superclap.alignment import align_textgrid_with_source_text

def load_item(id):

    # Text
    with open(id + ".txt", 'r') as file:
        text = file.read()

    # TextGrid
    tg = textgrid.TextGrid.fromFile(id + ".TextGrid")

    # Audio
    waveform = load_mono_audio(id + ".flac", config.audio.sample_rate)
    spec = spectogram(waveform, config.audio.n_fft, config.audio.n_mels, config.audio.hop_size, config.audio.win_size, config.audio.mel_norm, config.audio.mel_scale, config.audio.sample_rate)

    # Alignments
    al = align_textgrid_with_source_text(config, tg, text, spec.shape[1], id)
    if al is None:
        return None
    word_alignments, phoneme_alignments, combined_alignments = al

    # Results
    return waveform, spec.T, combined_alignments


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
                _, spec, alignments = it
                return spec, alignments
            else:
                print("Invalid item", id)

    return sample


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

    # Collate
    def collate(batch):
        specs, alignments = zip(*batch)

        # Pad specs
        max_len = max([s.shape[0] for s in specs])
        padded_specs = []
        for spec in specs:
            pad_size = max_len - spec.shape[0]
            padded_spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_size))
            padded_specs.append(padded_spec)
        specs = padded_specs

        return torch.stack(specs), alignments

    # Load loader
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, pin_memory = True, shuffle=False, collate_fn=collate)

    return loader