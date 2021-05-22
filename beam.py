from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from models.transformer import get_src_mask, get_tgt_mask
   

def beam(model, src_sent, beam_size, max_decoding_time_step, start_symbol, device):
    Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
    log_softmax = nn.LogSoftmax(dim=-1)
    src_embed = model.src_embeddings(src_sent)
    memory = model.encoder(model.pos_encoder(src_embed))

    hypotheses = [['[SOS]']]
    hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=device)
    completed_hypotheses = []

    t = 0
    while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
        t += 1
        ys = model.tokenizer.tgt_vocab.get_tensor(hypotheses, device)
        # ys = torch.ones(len(hypotheses), 1).fill_(start_symbol).type_as(src_sent.data)
        np_mask = torch.triu(torch.ones(ys.size(0), ys.size(0))==1).transpose(0,1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        tgt_embed = model.tgt_embeddings(ys)

        out = model.decoder(model.pos_encoder(tgt_embed), memory,
                           np_mask.to(device))
        out = model.out(out)
        prob = log_softmax(out.transpose(0,1)[:, -1])
        live_hyp_num = beam_size - len(completed_hypotheses)
        
        contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(prob) + prob).view(-1)
        top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

        prev_hyp_ids = top_cand_hyp_pos // len(model.tokenizer.tgt_vocab)
        hyp_word_ids = top_cand_hyp_pos % len(model.tokenizer.tgt_vocab)

        new_hypotheses = []
        live_hyp_ids = []
        new_hyp_scores = []

        for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
            prev_hyp_id = prev_hyp_id.item()
            hyp_word_id = hyp_word_id.item()
            cand_new_hyp_score = cand_new_hyp_score.item()

            hyp_word = model.tokenizer.tgt_vocab.id2word[hyp_word_id]
            new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]

            if hyp_word == '[EOS]':
                completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                        score=cand_new_hyp_score))
            else:
                new_hypotheses.append(new_hyp_sent)
                live_hyp_ids.append(prev_hyp_id)
                new_hyp_scores.append(cand_new_hyp_score)

        if len(completed_hypotheses) == beam_size:
            break

        live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=device)

        hypotheses = new_hypotheses
        hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=device)
    if len(completed_hypotheses) == 0:
        completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                score=hyp_scores[0].item()))

    completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
    return completed_hypotheses

class Translator(nn.Module):

    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs', 
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map', 
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))


    def _model_decode(self, trg_seq, enc_output, device):
        softmax = nn.Softmax(dim=-1)
        # tgt_mask = get_tgt_mask(trg_seq)
        output = self.model.decoder(trg_seq.transpose(0,1), enc_output.to(device)) 
        return softmax(self.model.out(output).transpose(0,1))

    def _get_init_state(self, src_seq, src_mask, device):
        beam_size = self.beam_size
        memory = self.model.encoder(src_seq.transpose(0,1), src_mask.to(device))
        dec_output = self._model_decode(self.init_seq, memory, device)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = memory.repeat(beam_size, 1, 1)

        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1
        
        beam_size = self.beam_size

        dec_output = dec_output[:, -1, :]

        index = (torch.arange(beam_size), gen_seq[:, step-2:step].transpose(0,1))
        min_val = torch.min(dec_output, dim=1).values
        dec_output.index_put_(index, min_val)
        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output.topk(beam_size)
        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)
 
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]
        
        # Copy the corresponding previous tokens.

        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]

        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, src_seq, device):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        self.model._reset_parameters()
        assert src_seq.size(0) == 1

        trg_eos_idx = self.trg_eos_idx 
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha 

        with torch.no_grad():
            src_mask = get_src_mask(src_seq, self.src_pad_idx)
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask, device)

            ans_idx = 0   # default
            for step in range(2, max_seq_len):    # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output.transpose(0,1), device)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)
                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx   
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()


def greedy_decode(model, src, max_len, tgt_sos_symbol, src_pad_token, tgt_eos_symbol, device):
    # model._reset_parameters()
    assert src.size(0) == 1
    trg_seq = torch.LongTensor([[tgt_sos_symbol]])
    softmax = nn.Softmax(dim=-1)
    model.eval()

    with torch.no_grad():
        for step in range(1, max_len):
            src_mask = get_src_mask(src, src_pad_token)
            memory = model.encoder(src.transpose(0,1), src_mask.to(device))

            output = model.decoder(trg_seq.transpose(0,1), memory.to(device)) 
            softmax_out =  softmax(model.out(output).transpose(0,1))
            softmax_out = softmax_out[:, -1, :]
            best_id = torch.argmax(softmax_out)
            trg_seq = torch.cat((trg_seq, best_id.view(-1, 1)), dim=-1)
    return trg_seq.squeeze().tolist()


class Beam:

    def __init__(self, beam_size=8, min_length=0, n_top=1, ranker=None,
                 start_token_id=2, end_token_id=3):
        self.beam_size = beam_size
        self.min_length = min_length
        self.ranker = ranker

        self.end_token_id = end_token_id
        self.top_sentence_ended = False

        self.prev_ks = []
        self.next_ys = [torch.LongTensor(beam_size).fill_(start_token_id)] # remove padding

        self.current_scores = torch.FloatTensor(beam_size).zero_()
        self.all_scores = []

        # The attentions (matrix) for each time.
        self.all_attentions = []

        self.finished = []

        # Time and k pair for finished.
        self.finished = []
        self.n_top = n_top

        self.ranker = ranker

    def advance(self, next_log_probs, current_attention):
        # next_probs : beam_size X vocab_size
        # current_attention: (target_seq_len=1, beam_size, source_seq_len)

        vocabulary_size = next_log_probs.size(1)
        # current_beam_size = next_log_probs.size(0)

        current_length = len(self.next_ys)
        if current_length < self.min_length:
            for beam_index in range(len(next_log_probs)):
                next_log_probs[beam_index][self.end_token_id] = -1e10

        if len(self.prev_ks) > 0:
            beam_scores = next_log_probs + self.current_scores.unsqueeze(1).expand_as(next_log_probs)
            # Don't let EOS have children.
            last_y = self.next_ys[-1]
            for beam_index in range(last_y.size(0)):
                if last_y[beam_index] == self.end_token_id:
                    beam_scores[beam_index] = -1e10 # -1e20 raises error when executing
        else:
            beam_scores = next_log_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        top_scores, top_score_ids = flat_beam_scores.topk(k=self.beam_size, dim=0, largest=True, sorted=True)

        self.current_scores = top_scores
        self.all_scores.append(self.current_scores)

        prev_k = top_score_ids / vocabulary_size  # (beam_size, )
        next_y = top_score_ids - prev_k * vocabulary_size  # (beam_size, )

        self.prev_ks.append(prev_k)
        self.next_ys.append(next_y)
        # for RNN, dim=1 and for transformer, dim=0.
        prev_attention = current_attention.index_select(dim=0, index=prev_k)  # (target_seq_len=1, beam_size, source_seq_len)
        self.all_attentions.append(prev_attention)


        for beam_index, last_token_id in enumerate(next_y):
            if last_token_id == self.end_token_id:
                # skip scoring
                self.finished.append((self.current_scores[beam_index], len(self.next_ys) - 1, beam_index))

        if next_y[0] == self.end_token_id:
            self.top_sentence_ended = True

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def done(self):
        return self.top_sentence_ended and len(self.finished) >= self.n_top

    def get_hypothesis(self, timestep, k):
        hypothesis, attentions = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hypothesis.append(self.next_ys[j + 1][k])
            # for RNN, [:, k, :], and for trnasformer, [k, :, :]
            attentions.append(self.all_attentions[j][k, :, :])
            k = self.prev_ks[j][k]
        attentions_tensor = torch.stack(attentions[::-1]).squeeze(1)  # (timestep, source_seq_len)
        return hypothesis[::-1], attentions_tensor

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                # global_scores = self.global_scorer.score(self, self.scores)
                # s = global_scores[i]
                s = self.current_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished = sorted(self.finished, key=lambda a: a[0], reverse=True)
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

class Predictor:

    def __init__(self, model, max_length=30, beam_size=8):
        # self.preprocess = preprocess
        # self.postprocess = postprocess
        self.model = model
        self.max_length = max_length
        self.beam_size = beam_size

    def predict_one(self, source_tensor, device, num_candidates=5):
        # source_preprocessed = self.preprocess(source)
        # source_tensor = torch.tensor(source_preprocessed).unsqueeze(0)  # why unsqueeze?
        # length_tensor = torch.tensor(len(source_preprocessed)).unsqueeze(0)

        sources_mask = get_src_mask(source_tensor, 0)
        memory = self.model.encoder(source_tensor, sources_mask)

        # decoder_state = self.model.decoder.init_decoder_state()
        # print('decoder_state src', decoder_state.src.shape)
        # print('previous_input previous_input', decoder_state.previous_input)
        # print('previous_input previous_layer_inputs ', decoder_state.previous_layer_inputs)


        # Repeat beam_size times
        memory_beam = memory.detach().repeat(1, self.beam_size, 1)  # (beam_size, seq_len, hidden_size)

        beam = Beam(beam_size=self.beam_size, min_length=0, n_top=num_candidates, ranker=None)

        for _ in range(self.max_length):

            new_inputs = beam.get_current_state().unsqueeze(1)  # (beam_size, seq_len=1)
            print(new_inputs.shape)
            memory_mask = get_tgt_mask(new_inputs)
            print(memory_mask.shape)
            print(memory_beam.shape)
            decoder_outputs = self.model.decoder(new_inputs.transpose(0,1), memory_beam,
                                                                            memory_mask)
            # decoder_outputs: (beam_size, target_seq_len=1, vocabulary_size)
            # attentions['std']: (target_seq_len=1, beam_size, source_seq_len)

            attention = self.model.decoder.decoder_layers[-1].memory_attention_layer.sublayer.attention
            beam.advance(decoder_outputs.squeeze(1), attention)

            if beam.done():
                break

        scores, ks = beam.sort_finished(minimum=num_candidates)
        hypothesises, attentions = [], []
        for i, (times, k) in enumerate(ks[:num_candidates]):
            hypothesis, attention = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)
            attentions.append(attention)

        self.attentions = attentions
        self.hypothesises = [[token.item() for token in h] for h in hypothesises]
        # hs = [self.postprocess(h) for h in self.hypothesises]
        exit()
        return list(reversed(hs))