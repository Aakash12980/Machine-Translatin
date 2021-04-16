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


class TransformerModel(nn.Module):
    def __init__(self, src_vocab_len, tgt_vocab_len, model_dim=512, n_heads=8, n_enc_layers=6, 
                n_dec_layers=6, fc_dim=2048, dropout=0.2, activation='relu'):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.tgt_vocab_len = tgt_vocab_len
        self.src_embeddings = nn.Embedding(src_vocab_len, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        enc_layer = nn.TransformerEncoderLayer(model_dim, n_heads, fc_dim, dropout, activation=activation)
        enc_norm = nn.LayerNorm(model_dim)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc_layers, enc_norm)
        self.tgt_embeddings = nn.Embedding(tgt_vocab_len, model_dim)
        dec_layer = nn.TransformerDecoderLayer(model_dim, n_heads, fc_dim, dropout, activation=activation)
        dec_norm = nn.LayerNorm(model_dim)
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec_layers, dec_norm)
        self.out = nn.Linear(512, tgt_vocab_len)
        self.init_weights()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        assert src.size(1) == tgt.size(1), "The number of source and target sentences should be equal."
        src = self.src_embeddings(src)
        src = self.pos_encoder(src)
        enc_encodings = self.encoder(src, src_mask)
        tgt = self.tgt_embeddings(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, tgt_mask)
        output = self.out(output)

        return output


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
