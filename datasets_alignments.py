import warnings
warnings.filterwarnings("ignore")
import torch
import torchaudio
import multiprocessing
import textgrid
from pathlib import Path
from tqdm import tqdm
from superclap.config import config
from superclap.audio import load_mono_audio, spectogram
from superclap.alignment import align_textgrid_with_source_text, extract_phonemes_in_words
from superclap.tokenizer import Tokenizer

def process_id(id):
    process_id = multiprocessing.current_process()._identity[0]
    device = "cuda:" + str(process_id % torch.cuda.device_count()) if torch.cuda.is_available() else "cpu"

    # Text
    with open("./external_datasets/" + id + ".txt", 'r') as file:
        text = file.read()

    # TextGrid
    tg = textgrid.TextGrid.fromFile("./external_datasets/" + id + ".TextGrid")

    # Audio
    waveform = load_mono_audio("./external_datasets/" + id + ".flac", config.audio.sample_rate).to(device)
    spec = spectogram(waveform, config.audio.n_fft, config.audio.n_mels, config.audio.hop_size, config.audio.win_size, config.audio.mel_norm, config.audio.mel_scale, config.audio.sample_rate)
    spec = spec.T

    # Alignments
    al = align_textgrid_with_source_text(config, tg, text, spec.shape[0], id)
    if al is None: # Some error during alignment
        return None
    word_alignments, phoneme_alignments, combined_alignments = al

    # Tokenize text
    tokenizer = get_tokenizer()
    tokens_lengths = []
    tokens_lengths_phonemes = []
    tokens = []
    for j in range(len(combined_alignments)):
        word, duration, real_world = combined_alignments[j][:3]
        encoded = tokenizer.encode(real_world)
        tokens += encoded
        tokens_lengths.append(len(encoded))
        if word is not None:
            tokens_lengths_phonemes.append(len(combined_alignments[j][3]))
        else:
            tokens_lengths_phonemes.append(1)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens_lengths = torch.tensor(tokens_lengths, dtype=torch.long)
    tokens_lengths_phonemes = torch.tensor(tokens_lengths_phonemes, dtype=torch.long)
    
    # Tokenize phonemes
    phonemes = []
    phonemes_durations = []
    for j in range(len(combined_alignments)):
        word, duration, real_world = combined_alignments[j][:3]
        if word is not None:
            for (phoneme, phoneme_duration) in combined_alignments[j][3]:
                phonemes.append(tokenizer.encode_phoneme(phoneme))
                phonemes_durations.append(phoneme_duration)
        else:
            phonemes.append(tokenizer.encode_phoneme("<SIL>"))
            phonemes_durations.append(duration)
    phonemes_durations = torch.tensor(phonemes_durations, dtype=torch.float)
    phonemes = torch.tensor(phonemes, dtype=torch.long)

    # Split audio to phoneme segments
    known_phonemes = set([])
    for (phoneme, start, end) in extract_phonemes_in_words(combined_alignments):
        if end - start > 80: # Ignore too long phonemes
            return None
        known_phonemes.add(phoneme)

    # Write
    Path("datasets/prepared/" + str(Path(id).parent)).mkdir(parents=True, exist_ok=True)
    torch.save(spec, "datasets/prepared/" + id + ".spec.pt")
    torch.save({ 
        "spec": spec, 
        "phonemes": phonemes, 
        "phonemes_duration": phonemes_durations,
        "tokens": tokens, 
        "len_tokens": tokens_lengths,
        "len_phonemes": tokens_lengths_phonemes
    }, "datasets/prepared/" + id + ".pt")

    return id, known_phonemes

def execute_run():

    # Load all text files
    ids = []
    for dataset in ["librilight-large-processed"]:
        with open("./external_datasets/" + dataset + "/files_valid.txt", 'r') as file:
            lines = file.readlines()
        ids += [(dataset + "/" + l.strip()) for l in lines]
    ids.sort()

    # Create directories
    Path("datasets/prepared").mkdir(parents=True, exist_ok=True)
    
    # Open output file
    multiprocessing.set_start_method('spawn')
    workers_count = torch.cuda.device_count() * 4 if torch.cuda.is_available() else 4
    indexes = {}
    with open("datasets/prepared.txt", "w") as tk:
        with multiprocessing.Manager() as manager:
            ids = manager.list(ids)
            with multiprocessing.Pool(processes=workers_count) as pool:
                for result in tqdm(pool.imap(process_id, ids, chunksize=32), total=len(ids)):
                    if result is not None:
                        id, phonemes = result

                        # Write to file list
                        tk.write(id + "\n")

                        # Write to index file
                        for p in phonemes:
                            if p not in indexes:
                                indexes[p] = [id]
                            else:
                                indexes[p].append(id)

    # Write indexes
    tokenizer = get_tokenizer()
    for p in indexes:
        p_index = tokenizer.encode_phoneme(p)
        with open("datasets/prepared/phoneme_" + str(p_index) + ".txt", "w") as file:
            for id in indexes[p]:
                file.write(id + "\n")
    
    # End
    print("Done")


tokenizer_cache = None
def get_tokenizer():
    global tokenizer_cache
    if tokenizer_cache is not None:
        return tokenizer_cache
    tokenizer_cache = Tokenizer()
    return tokenizer_cache

if __name__ == "__main__":
    execute_run()