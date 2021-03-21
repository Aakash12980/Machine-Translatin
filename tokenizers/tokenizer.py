from tokenizers import BertWordPieceTokenizer
import torch
import json
from itertools import chain
from collections import Counter
from nltk.tokenize import word_tokenize

class Tokenizer():
    def __init__(self, newa_vocab_path, eng_vocab_path):
        self.src_tokenizer = BertWordPieceTokenizer(newa_vocab_path, lowercase=True)
        self.tgt_tokenizer = BertWordPieceTokenizer(eng_vocab_path, lowercase=True)

    def encode(self, src_sents, tgt_sents=None, return_tensor=False):
        src_tokens = self.src_tokenizer.encode_batch(src_sents, return_tensor=return_tensor)
        if tgt_sents is not None:
            tgt_tokens = self.tgt_tokenizer.encode_batch(tgt_sents, return_tensor=return_tensor)

    def decode(self, src_ids, tgt_ids, return_tensor=False):
        pass

    @staticmethod
    def create_vocab(file_path, output_path, least_freq = 2):
        tokenizer = BertWordPieceTokenizer(clean_text=False, strip_accents=False, lowercase=True )
        files = [file_path]
        tokenizer.train(files, vocab_size=1000, min_frequency=least_freq, show_progress=True, 
                special_tokens=['[PAD]', '[UNK]', '[SOS]', '[EOS]'], limit_alphabet=1000, wordpieces_prefix="##")
        tokenizer.save(output_path)
        print(f"Vacabulary created at location {output_path}")

class SpaceVocab():
    def __init__(self, word2id=None):
        if word2id is not None:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['[PAD]'] = 0
            self.word2id['[UNK]'] = 1
            self.word2id['[SOS]'] = 2 
            self.word2id['[EOS]'] = 3
        self.unk_id = self.word2id['[UNK]']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __len__(self):
        return len(self.word2id)

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id
    
    def __repr__(self):
        return f'Vocabulary size = {len(self.word2id)}'

    def id2word(self, id):
        return self.id2word[id]

    def word2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def indices2word(self, word_ids):
        if type(word_ids[0]) == list:
            return [[self.id2word[w_id] for w_id in w_list] for w_list in word_ids]
        else:
            return [self.id2word[w_id] for w_id in word_ids]

    def get_token_id(self, sents, device, return_tensor=False):
        word_ids = self.word2indices(sents)
        if type(word_ids[0]) != list:
            if return_tensor:
                return torch.tensor(word_ids, dtype=torch.long, device=device)
            else:
                return word_ids
        else:
            sent_padded = []
            sents_padded = sents.copy()
            sent_len_list = [len(s) for s in sents]
            pad_token = self.word2id["[PAD]"]

            max_pad = max(sent_len_list)
            pad_token_list = [[pad_token]*i for i in range(max_pad)]
            [sents_padded[idx].extend(pad_token_list[max_pad-i]) for idx, i in enumerate(sent_len_list)]

            if return_tensor:
                return torch.t(torch.tensor(word_ids, dtype=torch.long, device=device))
            else:
                return sent_padded

class SpaceTokenizer():
    def __init__(self, newa_vocab_path, eng_vocab_path):
        self.src_tokenizer = SpaceVocab(json.load(open(newa_vocab_path, "r")))
        self.tgt_tokenizer = SpaceVocab(json.load(open(eng_vocab_path, "r")))
        
    def encode(self):
        pass

    def decode(self):
        pass
    
    @staticmethod
    def tokenize(sents, batch=True):
        tokens = []
        if batch:
            for s in sents:
                tokens.append(word_tokenize(s.lower().strip()))
        else: 
            tokens = word_tokenize(sents.lower().strip())
        return tokens        

    @staticmethod
    def create_vocab(file_path, output_path, least_freq = 2):
        vocab = SpaceVocab()
        sent_list = SpaceTokenizer.open_file(file_path)
        sent_tokens = SpaceTokenizer.tokenize(sent_list, batch=True)
        word_freq = Counter(chain(*sent_tokens))
        valid_words = [w for w, v in word_freq.items() if v >= least_freq]
        print(f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {least_freq}: {len(valid_words)}')
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)
        for word in top_k_words:
            vocab.add(word)
        json.dump(vocab.word2id, open(output_path, 'w'), indent=2)
        print(f"Vocabulary created at location {output_path}")

    @staticmethod
    def open_file(path):
        sents = []
        with open(path, "r", encoding="") as f:
            for s in f.readlines():
                sents.append(s.strip())
        return sents


if __name__ == "__main__":
    newa_file = "dataset/Newa_SS.txt"
    eng_file = "dataset/Eng_SS.txt"
    # vocab_newa_file = "wordpiece_vocab_files/vocab_newa.json"
    # vocab_eng_file = "wordpiece_vocab_files/vocab_eng.json"
    # Tokenizer.create_vocab(newa_file, vocab_newa_file)
    # Tokenizer.create_vocab(eng_file, vocab_eng_file)

    vocab_newa_file = "tokenizers/spacetoken_vocab_files/vocab_newa.json"
    vocab_eng_file = "tokenizers/spacetoken_vocab_files/vocab_eng.json"
    SpaceTokenizer.create_vocab(eng_file, vocab_eng_file)
    SpaceTokenizer.create_vocab(newa_file, vocab_newa_file)
