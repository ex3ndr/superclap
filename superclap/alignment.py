import re

def normalize_continious_phonemes(src):
    """
    Normalizing and fixing holes in phonemes
    """
    res = []
    time = 0
    for t in src:
        tok = t[0]
        start = t[1]
        end = t[2]
        if start != time:
            res.append(('<SIL>', time, start))
        res.append(t)
        time = end
    return res


def extract_textgrid_alignments(tg, index = 1):
    """
    Converts a TextGrid object to a list of tuples (phoneme, start, end)
    """
    output = []
    for t in tg[index]:
        ends = t.maxTime
        tok = t.mark
        if tok == '': # Ignore spaces
            continue
        if tok == 'spn':
            tok = '<UNK>'
        output.append((tok, t.minTime, t.maxTime))
    return output

def quantisize_phoneme_positions(src, phoneme_duration):
    """
    Quantisize phoneme positions, according to the single token duration
    """
    res = []
    for t in src:
        tok = t[0]
        # NOTE: We are expecting src to be normalized and start and end to match in adjacent tokens
        start = int(t[1] // phoneme_duration)
        end = int(t[2] // phoneme_duration)
        res.append((tok, start, end))
    return res


def continious_phonemes_to_discreete(raw_phonemes, phoneme_duration):
    """
    Convert continious phonemes to a list of integer intervals
    """

    # Normalize: add silence between intervals,
    #            ensure that start of any token is equal to end of a previous,
    #            ensure that first token is zero
    raw_phonemes = normalize_continious_phonemes(raw_phonemes)

    # Quantisize offsets: convert from real one to a discreete one
    quantisized = quantisize_phoneme_positions(raw_phonemes, phoneme_duration)

    # Convert to intervals
    intervals = [(i[0], i[2] - i[1]) for i in quantisized]

    return intervals


def compute_alignments(config, tg, total_duration):
    """
    Compute alignments from TextGrid object and style tensor
    """

    phoneme_duration = config.audio.hop_size / config.audio.sample_rate

    # Extract alignments
    x = extract_textgrid_alignments(tg)

    # Convert to discreete
    x = continious_phonemes_to_discreete(x, phoneme_duration)

    # Trim empty
    x = [i for i in x if i[1] > 0]

    # Pad with silence
    total_length = sum([i[1] for i in x])
    assert total_length <= total_duration # We don't have reversed case in our datasets
    if total_length < total_duration: # Pad with silence because textgrid is usually shorter
        x += [("<SIL>", total_duration - total_length)]

    return x

def align_textgrid_with_source_text(config, tg, text, total_duration, key):
    """
    Recombines textgrid phoneme and word alignments together and quantisize durations
    """

    # Load alignments
    word_alignments = extract_textgrid_alignments(tg, index = 0)
    phoneme_alignments = extract_textgrid_alignments(tg, index = 1)

    # Group phoneme alignments and word alignments
    combined_alignments = []
    for i in range(len(word_alignments)):
        word = word_alignments[i]
        phonemes = []
        for j in range(len(phoneme_alignments)):
            if phoneme_alignments[j][1] >= word[1] and phoneme_alignments[j][2] <= word[2]:
                phonemes.append(phoneme_alignments[j])
        combined_alignments.append((word, phonemes))

    # Split text to words
    pattern = r"[\w']+"
    def split_string(string):
        pattern = r"[\w']+"
        matches = re.finditer(pattern, string)
        result = []
        prev_end = 0

        for match in matches:
            start, end = match.span()
            if start > prev_end:
                result.append(string[prev_end:start])
            result.append(match.group())
            prev_end = end

        if prev_end < len(string):
            result.append(string[prev_end:])

        return result
    words = split_string(text)
    if re.match(r"[\w']+", words[0]):
        words = [" "] + words
    if re.match(r"[\w']+", words[-1]):
        words = words + [" "]

    # Normalize
    phoneme_duration = config.audio.hop_size / config.audio.sample_rate
    def quant(src):
        return int(src // phoneme_duration)
    normalized_combined = []
    time_q = 0
    w_i = 0
    for (word, word_start, word_end), phonemes in combined_alignments:
        word_start_q = quant(word_start)
        word_end_q = quant(word_end)

        # Add silence
        word_src = words[w_i]
        w_i += 1
        normalized_combined.append((None, word_start_q - time_q, word_src))
        
        # Append phonemes
        word_src = words[w_i]
        w_i += 1
        n_p = []
        ph_t = 0
        for phoneme, phoneme_start, phoneme_end in phonemes:
            ph_d = quant(phoneme_end) - quant(phoneme_start)
            n_p.append((phoneme, ph_d))
            ph_t += ph_d
        if ph_t != word_end_q - word_start_q:
            # print("Phoneme durations do not match word duration in " + key + ": " + str(ph_t) + ": " + str(word_end_q - word_start_q) + ": " + word_src)
            # print(phonemes)
            return None
        normalized_combined.append((word, word_end_q - word_start_q, word_src, n_p))
        
        # Adjust time
        time_q = word_end_q
    
    # Pad with silence
    word_src = words[w_i]
    w_i += 1
    total_length = sum([i[1] for i in normalized_combined])
    assert total_length <= total_duration # We don't have reversed case in our datasets
    if total_length < total_duration: # Pad with silence because textgrid is usually shorter
        normalized_combined += [(None, total_duration - total_length, word_src)]
    
    # Results
    return word_alignments, phoneme_alignments, normalized_combined
    
def extract_phonemes_in_words(combined):
    phonemes = []
    spec_offset = 0
    for segment in combined:
        word, duration, src = segment[:3]
        if word is not None:
            for (p, d) in segment[3]:
                phonemes.append((p, spec_offset, spec_offset + d))
                spec_offset += d
        else:
            spec_offset += duration
    return phonemes

def encode_text_and_align_with_phonemes(combined, tokenizer):
    tokens = []
    mapping = []
    bpe_offset = 0
    phoneme_offset = 0
    for segment in combined:
        word, duration, real_world = segment[:3]
        encoded = self.tokenizer.encode(real_world)

        # Append to tokens list
        tokens += encoded

        # Append to words list
        if word is not None:
            mapping.append(((bpe_offset, bpe_offset + len(encoded)), (phoneme_offset, phoneme_offset + len(segment[3]))))
            phoneme_offset += len(segment[3])
        else:
            phoneme_offset += 1

        # Update bpe offset
        bpe_offset += len(encoded)

    return tokens, mapping
    