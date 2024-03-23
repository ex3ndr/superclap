from .misc import dict_to_object

config = dict_to_object({
    
    # Shared audio parameters
    "audio": {
        "sample_rate": 24000,
        "n_mels": 100,
        "n_fft": 1024,
        "hop_size": 256,
        "win_size": 256 * 4,
        "mel_norm": "slaney",
        "mel_scale": "slaney",
        "norm_std": 2.2615,
        "norm_mean": -5.8843
    },

    # Text parameters
    "text": {
        "vocab_size": 4096,
        "phonemes": ['<UNK>', '<SIL>', '<BEGIN>', '<END>', 'a', 'ai˥˥', 'ai˥˩', 'ai˦', 'ai˧˥', 'ai˨', 'ai˨˩˦', 'ai˩', 'aj', 'aj˥', 'aj˥˦', 'aj˩', 'au˥˥', 'au˥˩', 'au˦', 'au˧˥', 'au˨', 'au˨˩˦', 'au˩', 'aw', 'aw˥', 'aw˥˦', 'aw˩', 'aː', 'aːɡ', 'aː˥', 'aː˥˦', 'aː˥˩', 'aː˦˥', 'aː˦˨', 'aː˧', 'aː˨ˀ˥', 'aː˨ˀ˦', 'aː˨˥', 'aː˨˦', 'aː˨˨', 'aː˨˨ˀ', 'aː˨˩', 'aː˨˩ˀ', 'aː˨˩˦', 'aː˨˩˨', 'aː˩', 'aː˩˩˦', 'a˥', 'a˥˥', 'a˥˦', 'a˥˧', 'a˥˩', 'a˦', 'a˦˥', 'a˦˨', 'a˧', 'a˧˥', 'a˧˩', 'a˨', 'a˨ˀ˥', 'a˨ˀ˦', 'a˨˥', 'a˨˦', 'a˨˨', 'a˨˨ˀ', 'a˨˩', 'a˨˩ˀ', 'a˨˩˦', 'a˨˩˨', 'a˩', 'a˩˩˦', 'b', 'bʲ', 'bʲː', 'bː', 'c', 'cʰ', 'cʷ', 'cʼ', 'cː', 'd', 'dz', 'dzʲ', 'dzʲː', 'dzː', 'dʐ', 'dʐː', 'dʑ', 'dʑː', 'dʒ', 'dʒː', 'dʲ', 'dʲː', 'dː', 'd̪', 'd̪z̪', 'd̪z̪ː', 'd̪ː', 'e', 'ei˥˥', 'ei˥˩', 'ei˦', 'ei˧˥', 'ei˨', 'ei˨˩˦', 'ei˩', 'ej', 'ew', 'eː', 'eː˥', 'eː˥˦', 'eː˥˧', 'eː˥˩', 'eː˦˥', 'eː˦˨', 'eː˧', 'eː˧˩', 'eː˨ˀ˥', 'eː˨ˀ˦', 'eː˨˥', 'eː˨˦', 'eː˨˨', 'eː˨˨ˀ', 'eː˨˩', 'eː˨˩ˀ', 'eː˨˩˦', 'eː˨˩˨', 'eː˩', 'eː˩˩˦', 'e˥', 'e˥˥', 'e˥˦', 'e˥˩', 'e˦', 'e˦˥', 'e˦˨', 'e˧', 'e˧˥', 'e˨', 'e˨ˀ˥', 'e˨˦', 'e˨˨', 'e˨˩', 'e˨˩ˀ', 'e˨˩˦', 'e˨˩˨', 'e˩', 'e˩˩˦', 'ẽ', 'f', 'fʲ', 'fʲː', 'fʷ', 'fː', 'h', 'hː', 'i', 'ia˥˩', 'ia˦˥', 'ia˧', 'ia˨˩', 'ia˩˩˦', 'iə˦˥', 'iə˦˨', 'iə˨ˀ˥', 'iə˨ˀ˦', 'iə˨˥', 'iə˨˦', 'iə˨˨', 'iə˨˨ˀ', 'iə˨˩', 'iə˨˩ˀ', 'iə˨˩˦', 'iə˨˩˨', 'iː', 'iː˥', 'iː˥˦', 'iː˥˧', 'iː˥˩', 'iː˦˥', 'iː˦˨', 'iː˧', 'iː˧˩', 'iː˨ˀ˥', 'iː˨ˀ˦', 'iː˨˥', 'iː˨˦', 'iː˨˨', 'iː˨˨ˀ', 'iː˨˩', 'iː˨˩ˀ', 'iː˨˩˦', 'iː˨˩˨', 'iː˩', 'iː˩˩˦', 'i˥˥', 'i˥˩', 'i˦', 'i˦˥', 'i˦˨', 'i˧', 'i˧˥', 'i˨', 'i˨ˀ˥', 'i˨ˀ˦', 'i˨˥', 'i˨˦', 'i˨˨', 'i˨˨ˀ', 'i˨˩', 'i˨˩ˀ', 'i˨˩˦', 'i˨˩˨', 'i˩', 'i˩˩˦', 'ĩ', 'i̥', 'j', 'jː', 'j̃', 'j̰', 'k', 'kp', 'kʰ', 'kʷ', 'kʷʼ', 'kʼ', 'kː', 'k̚', 'k͈', 'l', 'lː', 'l̩', 'm', 'mʲ', 'mʲː', 'mː', 'm̩', 'n', 'nː', 'n̩', 'n̪', 'n̪ː', 'o', 'ou˥˥', 'ou˥˩', 'ou˦', 'ou˧˥', 'ou˨', 'ou˨˩˦', 'ou˩', 'ow', 'oː', 'oː˥', 'oː˥˦', 'oː˥˧', 'oː˥˩', 'oː˦˥', 'oː˦˨', 'oː˧', 'oː˧˩', 'oː˨ˀ˥', 'oː˨ˀ˦', 'oː˨˥', 'oː˨˦', 'oː˨˨', 'oː˨˨ˀ', 'oː˨˩', 'oː˨˩ˀ', 'oː˨˩˦', 'oː˨˩˨', 'oː˩', 'oː˩˩˦', 'o˥', 'o˥˥', 'o˥˩', 'o˦', 'o˦˥', 'o˦˨', 'o˧', 'o˧˥', 'o˨', 'o˨ˀ˥', 'o˨ˀ˦', 'o˨˥', 'o˨˦', 'o˨˨', 'o˨˨ˀ', 'o˨˩', 'o˨˩ˀ', 'o˨˩˦', 'o˨˩˨', 'o˩', 'o˩˩˦', 'õ', 'p', 'pf', 'pʰ', 'pʲ', 'pʲː', 'pʷ', 'pː', 'p̚', 'p͈', 'r', 'rʲ', 'rʲː', 'rː', 'r̝', 'r̩', 'r̩ː˦˨', 'r̩ː˨˦', 'r̩˦˨', 'r̩˨˦', 's', 'sʰ', 'sʲ', 'sʲː', 'sʼ', 'sː', 's̪', 's̪ː', 's͈', 't', 'ts', 'tsʰ', 'tsʲ', 'tsʲː', 'tsː', 'tɕ', 'tɕʰ', 'tɕː', 'tɕ͈', 'tʂ', 'tʂː', 'tʃ', 'tʃʲ', 'tʃʲː', 'tʃː', 'tʰ', 'tʲ', 'tʲː', 'tʷ', 'tː', 't̚', 't̪', 't̪s̪', 't̪s̪ː', 't̪ʰ', 't̪ː', 't͈', 'u', 'ua˥˩', 'ua˦˥', 'ua˧', 'ua˨˩', 'ua˩˩˦', 'uə˦˥', 'uə˦˨', 'uə˨ˀ˥', 'uə˨ˀ˦', 'uə˨˥', 'uə˨˦', 'uə˨˨', 'uə˨˨ˀ', 'uə˨˩', 'uə˨˩ˀ', 'uə˨˩˦', 'uə˨˩˨', 'uː', 'uː˥', 'uː˥˦', 'uː˥˧', 'uː˥˩', 'uː˦˥', 'uː˦˨', 'uː˧', 'uː˧˩', 'uː˨ˀ˥', 'uː˨ˀ˦', 'uː˨˥', 'uː˨˦', 'uː˨˨', 'uː˨˨ˀ', 'uː˨˩', 'uː˨˩ˀ', 'uː˨˩˦', 'uː˨˩˨', 'uː˩', 'uː˩˩˦', 'u˥˥', 'u˥˩', 'u˦', 'u˦˥', 'u˦˨', 'u˧', 'u˧˥', 'u˨', 'u˨ˀ˥', 'u˨ˀ˦', 'u˨˥', 'u˨˦', 'u˨˨', 'u˨˨ˀ', 'u˨˩', 'u˨˩ˀ', 'u˨˩˦', 'u˨˩˨', 'u˩', 'u˩˩˦', 'ũ', 'v', 'vʲ', 'vʲː', 'vʷ', 'vː', 'w', 'w̃', 'x', 'xː', 'y', 'yː', 'yː˥˧', 'yː˥˩', 'yː˧˩', 'yː˩', 'y˥˥', 'y˥˩', 'y˦', 'y˧˥', 'y˨', 'y˨˩˦', 'y˩', 'z', 'zʲ', 'zʲː', 'z̩˥˥', 'z̩˥˩', 'z̩˦', 'z̩˧˥', 'z̩˨', 'z̩˨˩˦', 'z̩˩', 'z̪', 'z̪ː', 'æ', 'ç', 'çː', 'ð', 'ø', 'øː', 'øː˥˧', 'øː˥˩', 'øː˧˩', 'øː˩', 'ŋ', 'ŋm', 'ŋː', 'ŋ̍˧˥', 'œ', 'œ˥˩', 'œ˧˩', 'ɐ', 'ɐ̃', 'ɑ', 'ɑː', 'ɑː˥˧', 'ɑː˥˩', 'ɑː˧˩', 'ɑː˩', 'ɑ̃', 'ɒ', 'ɒː', 'ɓ', 'ɔ', 'ɔj', 'ɔʏ', 'ɔː', 'ɔː˥˩', 'ɔː˦˥', 'ɔː˦˨', 'ɔː˧', 'ɔː˨ˀ˥', 'ɔː˨ˀ˦', 'ɔː˨˥', 'ɔː˨˦', 'ɔː˨˨', 'ɔː˨˨ˀ', 'ɔː˨˩', 'ɔː˨˩ˀ', 'ɔː˨˩˦', 'ɔː˨˩˨', 'ɔː˩˩˦', 'ɔ˥', 'ɔ˥˧', 'ɔ˥˩', 'ɔ˦˥', 'ɔ˦˨', 'ɔ˧', 'ɔ˧˩', 'ɔ˨ˀ˥', 'ɔ˨ˀ˦', 'ɔ˨˥', 'ɔ˨˦', 'ɔ˨˨', 'ɔ˨˨ˀ', 'ɔ˨˩', 'ɔ˨˩ˀ', 'ɔ˨˩˦', 'ɔ˨˩˨', 'ɔ˩', 'ɔ˩˩˦', 'ɔ̃', 'ɕ', 'ɕʰ', 'ɕː', 'ɕ͈', 'ɖ', 'ɗ', 'ə', 'əw', 'əː˦˥', 'əː˦˨', 'əː˨ˀ˥', 'əː˨ˀ˦', 'əː˨˥', 'əː˨˦', 'əː˨˨', 'əː˨˨ˀ', 'əː˨˩', 'əː˨˩ˀ', 'əː˨˩˦', 'əː˨˩˨', 'ə˥', 'ə˥˥', 'ə˥˩', 'ə˦', 'ə˦˥', 'ə˦˨', 'ə˧˥', 'ə˨', 'ə˨ˀ˥', 'ə˨ˀ˦', 'ə˨˥', 'ə˨˦', 'ə˨˨', 'ə˨˨ˀ', 'ə˨˩', 'ə˨˩ˀ', 'ə˨˩˦', 'ə˨˩˨', 'ə˩', 'ɚ', 'ɛ', 'ɛː', 'ɛː˥˧', 'ɛː˥˩', 'ɛː˦˥', 'ɛː˦˨', 'ɛː˧', 'ɛː˧˩', 'ɛː˨ˀ˥', 'ɛː˨ˀ˦', 'ɛː˨˥', 'ɛː˨˦', 'ɛː˨˨', 'ɛː˨˨ˀ', 'ɛː˨˩', 'ɛː˨˩ˀ', 'ɛː˨˩˦', 'ɛː˨˩˨', 'ɛː˩˩˦', 'ɛ˥', 'ɛ˥˦', 'ɛ˥˧', 'ɛ˥˩', 'ɛ˦˥', 'ɛ˧', 'ɛ˧˩', 'ɛ˨˩', 'ɛ˩', 'ɛ˩˩˦', 'ɛ̃', 'ɜ', 'ɜː', 'ɝ', 'ɟ', 'ɟʝ', 'ɟʷ', 'ɟː', 'ɠ', 'ɡ', 'ɡb', 'ɡʷ', 'ɡː', 'ɣ', 'ɤ', 'ɤː˥˩', 'ɤː˦˥', 'ɤː˧', 'ɤː˨˩', 'ɤː˩˩˦', 'ɤ˥˩', 'ɤ˦˥', 'ɤ˧', 'ɤ˨˩', 'ɤ˩˩˦', 'ɥ', 'ɦ', 'ɦː', 'ɧ', 'ɨ', 'ɨə˦˥', 'ɨə˦˨', 'ɨə˨ˀ˥', 'ɨə˨ˀ˦', 'ɨə˨˥', 'ɨə˨˦', 'ɨə˨˨', 'ɨə˨˨ˀ', 'ɨə˨˩', 'ɨə˨˩ˀ', 'ɨə˨˩˦', 'ɨə˨˩˨', 'ɨː', 'ɨː˦˥', 'ɨː˦˨', 'ɨː˨ˀ˥', 'ɨː˨ˀ˦', 'ɨː˨˥', 'ɨː˨˦', 'ɨː˨˨', 'ɨː˨˨ˀ', 'ɨː˨˩', 'ɨː˨˩ˀ', 'ɨː˨˩˦', 'ɨː˨˩˨', 'ɨ˦˥', 'ɨ˦˨', 'ɨ˨ˀ˥', 'ɨ˨ˀ˦', 'ɨ˨˥', 'ɨ˨˦', 'ɨ˨˨', 'ɨ˨˨ˀ', 'ɨ˨˩', 'ɨ˨˩ˀ', 'ɨ˨˩˦', 'ɨ˨˩˨', 'ɨ̥', 'ɪ', 'ɪː', 'ɪ˥', 'ɪ˥˦', 'ɪ˥˧', 'ɪ˥˩', 'ɪ˧˩', 'ɪ˩', 'ɫ', 'ɫː', 'ɫ̩', 'ɭ', 'ɭː', 'ɯ', 'ɯa˥˩', 'ɯa˦˥', 'ɯa˧', 'ɯa˨˩', 'ɯa˩˩˦', 'ɯː', 'ɯː˥˩', 'ɯː˦˥', 'ɯː˧', 'ɯː˨˩', 'ɯː˩˩˦', 'ɯ˥˩', 'ɯ˦˥', 'ɯ˧', 'ɯ˨˩', 'ɯ˩˩˦', 'ɯ̥', 'ɰ', 'ɰ̃', 'ɱ', 'ɲ', 'ɲː', 'ɳ', 'ɳː', 'ɴ', 'ɴː', 'ɵ', 'ɵ˥˧', 'ɵ˥˩', 'ɵ˧˩', 'ɵ˩', 'ɸ', 'ɸʲ', 'ɸʲː', 'ɸː', 'ɹ', 'ɻ', 'ɽ', 'ɾ', 'ɾʲ', 'ɾʲː', 'ɾː', 'ɾ̃', 'ʁ', 'ʂ', 'ʂː', 'ʃ', 'ʃʲ', 'ʃʲː', 'ʃː', 'ʄ', 'ʈ', 'ʈʂ', 'ʈʂʰ', 'ʈʰ', 'ʈʲ', 'ʈʷ', 'ʈː', 'ʉ', 'ʉː', 'ʉː˥˧', 'ʉː˥˩', 'ʉː˧˩', 'ʊ', 'ʊ˥', 'ʊ˥˦', 'ʊ˥˩', 'ʊ˧˩', 'ʊ˩', 'ʋ', 'ʋʲ', 'ʋʲː', 'ʋː', 'ʌ', 'ʌː', 'ʎ', 'ʎː', 'ʏ', 'ʏ˥˧', 'ʏ˥˩', 'ʏ˧˩', 'ʏ˩', 'ʐ', 'ʐː', 'ʐ̩˥˥', 'ʐ̩˥˩', 'ʐ̩˦', 'ʐ̩˧˥', 'ʐ̩˨', 'ʐ̩˨˩˦', 'ʐ̩˩', 'ʑ', 'ʑː', 'ʒ', 'ʒʲ', 'ʒʲː', 'ʔ', 'ʝ', 'β', 'θ']
    },

    # Text Encoder
    "text_encoder": {
        "n_embeddings": 256,
        "n_heads": 4,
        "n_layers": 4,
        "n_dim": 256,
        "n_dim_head": 32,
        "n_dim_ffn": 1024,
    }
})