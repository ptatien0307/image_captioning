import torch
from torch.nn import Module, Linear
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Attention(Module):
    def __init__(self, attention_dim, encoder_dim, decoder_dim):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim
        self.W_layer = Linear(decoder_dim, attention_dim).to(device)
        self.U_layer = Linear(encoder_dim, attention_dim).to(device)
        self.V_layer = Linear(attention_dim, 1).to(device)

    def forward(self, keys, query):
        U = self.U_layer(keys)     # (batch_size, num_layers, attention_dim)
        W = self.W_layer(query) # (batch_size, attention_dim)

        combined = torch.tanh(U + W.unsqueeze(1)) # (batch_size, num_layers, attention_dim)
        score = self.V_layer(combined)  # (batch_size, num_layers, 1)
        score = score.squeeze(2) # (batch_size, num_layers)

        weights = F.softmax(score, dim=1)    # (batch_size, num_layers)

        context = keys * weights.unsqueeze(2) # (batch_size, num_layers, feature_dim)
        context = context.sum(dim=1)   # (batch_size, feature_dim)
        return context, weights
