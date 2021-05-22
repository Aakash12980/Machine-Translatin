from tokenizers import BertWordPieceTokenizer
import torch
import json
from itertools import chain
from collections import Counter
from nltk.tokenize import word_tokenize
from utils import *
import utils
from beam import *

class BertTokenizer():
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

    def word2indices(self, sents, is_src):
        if not isinstance(sents, str):
            if is_src:
                return [[self[w] for w in word_tokenize(s)] for s in sents]
            else:
                return [[self[w] for w in ["[SOS]"] + word_tokenize(s) +["[EOS]"]] for s in sents]
        else:
            if is_src:
                return [self[w] for w in word_tokenize(sents)]
            else:
                return [self[w] for w in ["[SOS]"] + word_tokenize(sents) + ["[EOS]"]]

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

    def get_token_id(self, sents, device, is_src, return_tensor=False):
        word_ids = self.word2indices(sents, is_src)
        if not isinstance(word_ids[0], list):
            if return_tensor:
                return torch.tensor(word_ids, dtype=torch.long, device=device)
            else:
                return word_ids
        else:
            sent_padded = []
            sents_padded = word_ids.copy()
            sent_len_list = [len(s) for s in word_ids]
            pad_token = self.word2id["[PAD]"]

            max_pad = max(sent_len_list)
            pad_token_list = [[pad_token]*i for i in range(max_pad)]
            [sents_padded[idx].extend(pad_token_list[max_pad-i]) for idx, i in enumerate(sent_len_list)]

            if return_tensor:
                return torch.t(torch.tensor(word_ids, dtype=torch.long, device=device)), sent_len_list
            else:
                return sent_padded, sent_len_list

class SpaceTokenizer():
    def __init__(self, newa_vocab_path, eng_vocab_path):
        self.src_vocab = SpaceVocab(json.load(open(newa_vocab_path, "r", encoding="utf8")))
        self.tgt_vocab = SpaceVocab(json.load(open(eng_vocab_path, "r", encoding="utf8")))
    
    def encode(self, batch, device, return_tensor=False):
        if isinstance(batch[0], str):
            src_tensor, src_len = self.src_vocab.get_token_id(batch, device, is_src=True, return_tensor=return_tensor)
            return src_tensor, src_len
        else:
            src_tensor, src_len = self.src_vocab.get_token_id(batch[0], device, is_src=True, return_tensor=return_tensor)
            tgt_tensor, tgt_len = self.tgt_vocab.get_token_id(batch[1], device, is_src=False, return_tensor=return_tensor)

            return src_tensor, tgt_tensor, src_len, tgt_len

    @staticmethod
    def decode(model, src_sent_list, beam_size, max_decoding_time_step, device):
        was_training = model.training
        model.eval()

        hypotheses = []
        with torch.no_grad():
            for src_sent in src_sent_list:
                # example_hyps = utils.beam_search(model, src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step, device=device)
                example_hyps = beam(model, src_sent, beam_size, max_decoding_time_step, 2, device)
                hypotheses.append(example_hyps)
                # print(hypotheses)
        if was_training: model.train(was_training)

        return hypotheses
    
    @staticmethod
    def tokenize(sents, batch=True, wrap_inner_list = False):
        tokens = []
        if batch:
            for s in sents:
                if wrap_inner_list:
                    tokens.append([word_tokenize(s.lower().strip())])
                else:
                    tokens.append(word_tokenize(s.lower().strip()))
        else:
            tokens = word_tokenize(sents.lower().strip())
        return tokens        

    @staticmethod
    def create_vocab(file_path, output_path, least_freq = 2):
        vocab = SpaceVocab()
        sent_list = utils.open_file(file_path)
        sent_tokens = SpaceTokenizer.tokenize(sent_list, batch=True)
        word_freq = Counter(chain(*sent_tokens))
        valid_words = [w for w, v in word_freq.items() if v >= least_freq]
        print(f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {least_freq}: {len(valid_words)}')
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)
        for word in top_k_words:
            vocab.add(word)
        json.dump(vocab.word2id, open(output_path, 'w', encoding="utf8"), ensure_ascii=False, indent=2)
        print(f"Vocabulary created at location {output_path}")

if __name__ == "__main__":
    # tokenizer = SpaceTokenizer("./NMTtokenizers/spacetoken_vocab_files/vocab_newa.json", 
    #             "./NMTtokenizers/spacetoken_vocab_files/vocab_eng.json"
    #             )

    # batch = [["छिं नयेगु नसा म्हसिइका दिसँ ", "छि च्वनेगु थाय् म्हसिइका दिसँ ", "छिं पुनेगु वस: म्हसिइका दिसँ"], ["I am a student.", "Hey you!", "yess boy, I am."]]
    # src, tgt, src_len, tgt_len = tokenizer.encode(batch, "cpu", True)
    # print(src)
    # print(tgt)
    # print(src.shape)
    # print(tgt.shape)


    newa_file = "dataset/src_train.txt"
    eng_file = "dataset/tgt_train.txt"
    # vocab_newa_file = "wordpiece_vocab_files/vocab_newa.json"
    # vocab_eng_file = "wordpiece_vocab_files/vocab_eng.json"
    # BertTokenizer.create_vocab(newa_file, vocab_newa_file)
    # BertTokenizer.create_vocab(eng_file, vocab_eng_file)

    vocab_newa_file = "tokenizers/spacetoken_vocab_files/vocab_newa.json"
    vocab_eng_file = "tokenizers/spacetoken_vocab_files/vocab_eng.json"
    SpaceTokenizer.create_vocab(eng_file, vocab_eng_file)
    SpaceTokenizer.create_vocab(newa_file, vocab_newa_file)

