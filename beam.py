from collections import namedtuple
import torch
import torch.nn as nn
from models.transformer import get_src_mask, get_tgt_mask

class Translator(nn.Module):

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


def beam_search_transformer(model, src_tensor, beam_size, max_decoding_time_step, src_pad_idx, eos_id, device):
    Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

    log_softmax = nn.LogSoftmax(dim=-1)
    src_mask = get_src_mask(src_tensor, src_pad_idx)
    memory = model.encoder(src_tensor.transpose(0,1), src_mask.to(device))

    hypotheses = [['[SOS]']]
    hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=device) 
    completed_hypotheses = []

    t = 0
    while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
        t += 1
        hyp_num = len(hypotheses)

        trg_seq = torch.tensor([model.tokenizer.tgt_vocab[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=device)
        output = model.decoder(trg_seq.view(-1,1), memory.to(device)) 

        # log probabilities over target words
        log_p_t = log_softmax(model.out(output).transpose(0,1))

        live_hyp_num = beam_size - len(completed_hypotheses)
        contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
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

        hypotheses = new_hypotheses
        hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=device)

    if len(completed_hypotheses) == 0:
        completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                score=hyp_scores[0].item()))

    completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

    return completed_hypotheses
