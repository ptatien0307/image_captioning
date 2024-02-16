import math
import torch
from torch.nn import Module, Dropout

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PositionalEncoding(Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 50):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(1, max_len, d_model).to(device)
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
