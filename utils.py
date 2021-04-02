import enum
from nltk.translate import bleu_score
import torch
import torch.nn as nn
import shutil
import sys
from collections import namedtuple
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from NMTtokenizers.tokenizer import SpaceTokenizer

def open_file(path):
    sents = []
    with open(path, encoding='utf8') as f:
        text = f.read().splitlines()
        for s in text:
            sents.append(s.strip())
    try:
        return sents[:100]
    except:
        return sents

def compute_bleu_score(output, labels):
    refs = SpaceTokenizer.tokenize(labels, batch=True, wrap_inner_list = True)
    output_tokens = SpaceTokenizer.tokenize(output, batch=True)
    weights = (1.0/2.0, 1.0/2.0, )
    score = corpus_bleu(refs, output_tokens, smoothing_function=SmoothingFunction(epsilon=1e-10).method1, weights=weights)
    return score

def evaluate(model, data_loader, e_loss, device, batch_size, beam_size, max_decoding_time_step):
    model.eval()
    eval_loss = e_loss
    total_steps = 0
    bleu_score = 0

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            src_tensor, tgt_tensor, src_len, tgt_len = model.tokenizer.encode(batch, device, return_tensor=True)
            loss = -model(src_tensor, tgt_tensor, src_len, model.tokenizer.tgt_vocab.word2id['[UNK]'], device)

            batch_loss = loss.sum()
            avg_loss = batch_loss/batch_size
            eval_loss += avg_loss.item()
            total_steps = step

            hypotheses = SpaceTokenizer.decode(model, batch[0],
                             beam_size=beam_size,
                             max_decoding_time_step=max_decoding_time_step, device=device)

            output = []
            for hyps in hypotheses:
                top_hyp = hyps[0]
                output.append(' '.join(top_hyp.value))

            score = compute_bleu_score(output, batch[1])
            bleu_score += score
    model.train()

    return eval_loss/total_steps, bleu_score/total_steps

def save_model_checkpt(state, is_best, check_pt_path, best_model_path):
    f_path = check_pt_path
    torch.save(state, f_path)

    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_checkpt(model, checkpt_path, device, optimizer=None):
    checkpoint = torch.load(checkpt_path, map_location=device)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    eval_loss = checkpoint["eval_loss"]
    epoch = checkpoint["epoch"]

    return model, optimizer, eval_loss, epoch

def beam_search(model, src_sent, beam_size, max_decoding_time_step, device):
    Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
    src_sents_var, sent_len = model.tokenizer.encode([src_sent], device, return_tensor=True)
    log_softmax = nn.LogSoftmax(dim=-1)

    src_encodings, hidden_state, cell_state = model.encoder(src_sents_var, sent_len)
    src_encodings_att_linear = model.decoder.attention(src_encodings)

    h_tm1 = (hidden_state, cell_state )
    att_tm1 = torch.zeros(1, model.hidden_size, device=device)

    eos_id = model.tokenizer.tgt_vocab.word2id['[EOS]']

    hypotheses = [['[SOS]']]
    hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=device)
    completed_hypotheses = []

    t = 0
    while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
        t += 1
        hyp_num = len(hypotheses)

        exp_src_encodings = src_encodings.expand(hyp_num,
                                                    src_encodings.size(1),
                                                    src_encodings.size(2))

        exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                        src_encodings_att_linear.size(1),
                                                                        src_encodings_att_linear.size(2))

        y_tm1 = torch.tensor([model.tokenizer.tgt_vocab[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=device)
        y_t_embed = model.decoder.model_embeddings(y_tm1)

        x = torch.cat([y_t_embed, att_tm1], dim=-1) 

        (h_t, cell_t), att_t, _  = model.decoder.step(x, h_tm1,
                                                    exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

        # log probabilities over target words
        log_p_t = log_softmax(model.vocab_fc(att_t))

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

        live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=device)
        h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
        att_tm1 = att_t[live_hyp_ids]

        hypotheses = new_hypotheses
        hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=device)

    if len(completed_hypotheses) == 0:
        completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                score=hyp_scores[0].item()))

    completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

    return completed_hypotheses
    
    