import torch
import torch.nn.functional as F
from torch.nn import Module, Linear

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Attention(Module):
    """
        Attention mechanism module for sequence-to-sequence models.

        This class implements the attention mechanism used in sequence-to-sequence models,
        which helps the model focus on relevant parts of the input sequence when generating
        the output sequence.

        Args:
            attention_dim (int): Dimensionality of the attention mechanism.
            encoder_dim (int): Dimensionality of the encoder's output.
            decoder_dim (int): Dimensionality of the decoder's output.

        Attributes:
            attention_dim (int): Dimensionality of the attention mechanism.
            W_layer: Linear layer to transform decoder's output.
            U_layer: Linear layer to transform encoder's output.
            V_layer: Linear layer to compute attention scores.
    """
    def __init__(self, attention_dim, encoder_dim, decoder_dim):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim
        self.W_layer = Linear(decoder_dim, attention_dim).to(device)
        self.U_layer = Linear(encoder_dim, attention_dim).to(device)
        self.V_layer = Linear(attention_dim, 1).to(device)

    def forward(self, keys, query):
        U = self.U_layer(keys)                          # (batch_size, num_layers, attention_dim)
        W = self.W_layer(query)                         # (batch_size, attention_dim)

        combined = torch.tanh(U + W.unsqueeze(1))       # (batch_size, num_layers, attention_dim)
        score = self.V_layer(combined)                  # (batch_size, num_layers, 1)
        score = score.squeeze(2)                        # (batch_size, num_layers)

        weights = F.softmax(score, dim=1)               # (batch_size, num_layers)

        context = keys * weights.unsqueeze(2)           # (batch_size, num_layers, feature_dim)
        context = context.sum(dim=1)                    # (batch_size, feature_dim)
        return context, weights
