import torch
import torch.nn as nn
from torch.nn.modules import dropout
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, dropout_rate=0.2, n_layers=5):
        super(Encoder, self).__init__()
        self.model_embeddings = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bias=True, bidirectional=True, dropout=dropout_rate, num_layers=n_layers)
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, input_ids, src_length):
        embeddings = self.model_embeddings(input_ids)
        padded_seq = pack_padded_sequence(embeddings, src_length)
        enc_hiddens, (last_hidden, last_cell) = self.lstm(padded_seq)
        enc_hiddens = pad_packed_sequence(sequence=enc_hiddens)[0].permute(1,0,2)
        enc_hidden_combined = self.fc_hidden(torch.cat((last_hidden[0], last_hidden[1]), 1))
        enc_cell_combined = self.fc_cell(torch.cat((last_cell[0], last_cell[1]), 1))

        return enc_hiddens, enc_hidden_combined, enc_cell_combined

class Decoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, dropout_rate=0.2, n_layers=5):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.model_embeddings = nn.Embedding(input_size, embed_size)
        self.lstm_cell = nn.LSTMCell(embed_size + hidden_size, hidden_size, bias=True)
        self.combined_op_fc = nn.Linear(hidden_size*3, hidden_size, bias=False)
        self.attention = nn.Linear(hidden_size*2, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, tgt_token_ids, enc_hidden, enc_mask, dec_init_hidden, dec_init_cell):
        combined_op = []
        batch_size = enc_hidden.size(0)
        tgt_token_ids = tgt_token_ids[:-1]
        dec_init_state = (dec_init_hidden, dec_init_cell)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        enc_hiddens_proj = self.attention(enc_hidden)
        y = self.model_embeddings(tgt_token_ids)

        for y_t in torch.split(y, 1, dim=0):
            y_t = torch.squeeze(y_t, 0)
            ybar_t = torch.cat((y_t, o_prev), 1)
            _, o_t, _ = self.step(ybar_t, dec_init_state, enc_hidden, enc_hiddens_proj, enc_mask)
            combined_op.append(o_t)
            o_prev = o_t

        combined_outputs = torch.stack(combined_op, dim=0)
        return combined_outputs

    def step(self, Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks):

        combined_output = None

        dec_state = self.lstm_cell(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state
        e_t = torch.squeeze(torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, dim=2)), dim=2)

        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        soft_max = nn.Softmax(dim=1)
        alpha_t = soft_max(e_t)
        a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t, dim=1), enc_hiddens), dim=1)
        U_t = torch.cat((dec_hidden, a_t), 1)
        V_t = self.combined_op_fc(U_t)

        m = nn.Tanh()
        O_t = self.dropout(m(V_t))

        combined_output = O_t
        return dec_state, combined_output, e_t

class Seq2Seq(nn.Module):
    def __init__(self, embed_size, hidden_size, src_vocab_len, tokenizer, dropout_rate=0.2, n_layers=5):
        super(Seq2Seq, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.tokenizer = tokenizer

        self.encoder = Encoder(embed_size, hidden_size, dropout_rate, n_layers)
        self.decoder = Decoder(embed_size, hidden_size, dropout_rate, n_layers)
        self.combined_output = nn.Linear(hidden_size*3, hidden_size, bias=False)
        self.logSoftmax = nn.LogSoftmax(dim=-1)
        self.vocab_fc = nn.Linear(hidden_size, src_vocab_len, bias=False)

    def forward(self, src_token_ids, tgt_token_ids, src_len, tgt_len, tgt_pad_id):
        enc_hiddens, dec_hidden_state, dec_cell_state = self.encoder(src_token_ids, src_len)
        enc_masks = self.mask_sent(enc_hiddens, src_len)
        combined_op = self.decoder(tgt_token_ids, enc_hiddens, enc_masks, dec_hidden_state, dec_cell_state)
        P = self.logSoftmax(self.vocab_fc(combined_op))

        tgt_masks = (tgt_token_ids != tgt_pad_id).float()

        tgt_words_log_prob = torch.gather(P, index=tgt_token_ids[1:].unsqueeze(-1), dim=-1).squeeze(-1) * tgt_masks[1:]
        scores = tgt_words_log_prob.sum(dim=0)
        return scores

    def mask_sent(self, enc_hiddens, src_len):
        enc_mask = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for id, src_len in enumerate(src_len):
            enc_mask[id, src_len:] = 1
        return enc_mask.to(self.device)



