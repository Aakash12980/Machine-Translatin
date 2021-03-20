from tokenizers import BertWordPieceTokenizer

class Tokenizer():
    def __init__(self, newa_vocab_path, eng_vocab_path):
        self.src_tokenizer = BertWordPieceTokenizer(newa_vocab_path, lowercase=True)
        self.tgt_tokenizer = BertWordPieceTokenizer(eng_vocab_path, lowercase=True)

    def encode(self, src_sents, tgt_sents=None):
        pass

    def decode(self, src_ids, tgt_ids):
        pass

    @staticmethod
    def create_vocab(file_path, output_path):
        tokenizer = BertWordPieceTokenizer(clean_text=False, strip_accents=False, lowercase=True )
        files = [file_path]
        tokenizer.train(files, vocab_size=1000, min_frequency=2, show_progress=True, 
                special_tokens=['[PAD]', '[UNK]', '[SOS]', '[EOS]'], limit_alphabet=1000, wordpieces_prefix="##")
        tokenizer.save(output_path)
        print("Vacabulary created successfully.")
        
if __name__ == "__main__":
    newa_file = "dataset/Newa_SS.txt"
    eng_file = "dataset/Eng_SS.txt"
    vocab_newa_file = "vocab_files/vocab_newa.json"
    vocab_eng_file = "vocab_files/vocab_eng.json"
    Tokenizer.create_vocab(newa_file, vocab_newa_file)
    Tokenizer.create_vocab(eng_file, vocab_eng_file)
