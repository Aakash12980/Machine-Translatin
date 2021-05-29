import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout_rate=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.model_dim = model_dim
        self.dropout = nn.Dropout(dropout_rate)
        pos_enc = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).view(-1,1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, inputs):
        inputs = inputs * math.sqrt(self.model_dim)
        inputs = inputs + self.pos_enc[:inputs.size(0), :]
        return self.dropout(inputs)

class Encoder(nn.Module):
    def __init__(self, src_vocab_len, model_dim, fc_dim, n_heads, n_enc_layers, pad_idx, dropout, activation):
        super(Encoder, self).__init__()
        self.src_embeddings = nn.Embedding(src_vocab_len, model_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        enc_layer = nn.TransformerEncoderLayer(model_dim, n_heads, fc_dim, dropout, activation=activation)
        enc_norm = nn.LayerNorm(model_dim)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc_layers, enc_norm)

    def forward(self, src, src_mask):
        src = self.src_embeddings(src)
        src = self.pos_encoder(src)
        # return self.encoder(src, src_key_padding_mask=src_mask) #src_mask is giving error.
        return self.encoder(src)
        
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_len, model_dim, fc_dim, n_heads, n_dec_layers, pad_idx, dropout, activation):
        super(Decoder, self).__init__()
        self.tgt_embeddings = nn.Embedding(tgt_vocab_len, model_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        dec_layer = nn.TransformerDecoderLayer(model_dim, n_heads, fc_dim, dropout, activation=activation)
        dec_norm = nn.LayerNorm(model_dim)
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec_layers, dec_norm)

    def forward(self, tgt, enc_encodings, tgt_mask=None):
        tgt = self.tgt_embeddings(tgt)
        tgt = self.pos_encoder(tgt)
        if tgt_mask is None:
            output = self.decoder(tgt, enc_encodings, tgt_mask)
        else:
            output = self.decoder(tgt, enc_encodings, tgt_mask)
        return output

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_len, tgt_vocab_len, tokenizer, model_dim=512, n_heads=8, n_enc_layers=6, 
                n_dec_layers=6, fc_dim=2048, dropout=0.2, activation='relu'):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.tokenizer = tokenizer
        self.tgt_vocab_len = tgt_vocab_len
        self.encoder = Encoder(src_vocab_len, model_dim, fc_dim, n_heads, n_enc_layers, self.tokenizer.src_vocab["[PAD]"], dropout, activation)
        self.decoder = Decoder(tgt_vocab_len, model_dim, fc_dim, n_heads, n_dec_layers, self.tokenizer.tgt_vocab["[PAD]"], dropout, activation)
        self.out = nn.Linear(model_dim, tgt_vocab_len)

    def forward(self, src, tgt, device):
        # src, tgt have shape (batch, seq_len)
        assert src.size(0) == tgt.size(0), "The batch size of source and target sentences should be equal."
        src_mask = get_src_mask(src, self.tokenizer.src_vocab["[PAD]"])
        tgt_mask = get_tgt_mask(tgt)
        enc_encodings = self.encoder(src.transpose(0,1), src_mask.to(device)) 
        output = self.decoder(tgt.transpose(0,1), enc_encodings, tgt_mask.to(device))
        output = self.out(output)
        return output.transpose(0,1).contiguous().view(-1, output.size(-1))

def get_src_mask(src_tensor, src_pad_id):
    mask = src_tensor != src_pad_id   
    return mask

def get_tgt_mask(tgt_tensor):
    seq_len = tgt_tensor.size(-1)
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
