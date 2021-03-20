import torch
import torch.nn as nn
from torch.nn.modules import dropout

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, bias=True, dropout=dropout_rate, bidirectional=True)

        self.decoder = nn.GRUCell(hidden_size, hidden_size)
