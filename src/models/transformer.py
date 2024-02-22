import math
import torch
from torch.nn import Module, Linear, Embedding, TransformerDecoder, TransformerDecoderLayer, Transformer
from .modules.base import TransformerEncoder
from .modules.pos import PositionalEncoding
# from .modules import PositionalEncoding, TransformerEncoder

class Decoder(Module):

    def __init__(self, n_tokens, d_model, n_heads, dim_forward,
                 n_layers, dropout = 0.2):
        super(Decoder, self).__init__()
        self.embedding = Embedding(n_tokens, d_model) # embedding layer
        self.pos_encoder = PositionalEncoding(d_model, dropout) # positional encoder

        decoder_layers = TransformerDecoderLayer(d_model, n_heads, dim_forward, dropout, batch_first=True) # encoder layer
        self.transformer_decoder = TransformerDecoder(decoder_layers, n_layers) # transformer encoder


        self.d_model = d_model # number of features
        self.linear = Linear(d_model, n_tokens) # last linear model for prediction

    def forward(self, features, captions, padding_mask, captions_mask = None):
        """
        Arguments:
            captions: Tensor, shape ``[batch_size, seq_len]``
            captions_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[batch_size, seq_len, n_tokens]``
        """
        captions = self.embedding(captions)
        captions = captions * math.sqrt(self.d_model)
        captions = self.pos_encoder(captions)

        if captions_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            captions_mask = Transformer.generate_square_subsequent_mask(captions.size(1))

        output = self.transformer_decoder(tgt=captions,
                                          memory=features,
                                          tgt_key_padding_mask=padding_mask,
                                          tgt_mask=captions_mask)
        output = self.linear(output)
        return output

    def predict(self, feature, max_length, vocab):
        word = torch.tensor([vocab.word2index['<SOS>']] + [0] * (max_length - 1)).view(1, -1)
        padding_mask = torch.Tensor([True] * max_length).view(1, -1)

        predicted_captions = []

        for i in range(max_length - 1):
            # Update the padding masks
            padding_mask[:, i] = False

            # Get the model prediction for the next word
            output = self.forward(feature, word, padding_mask)
            output = output[0, i]
            predicted_word_idx = output.argmax(dim=-1)
            predicted_captions.append(predicted_word_idx.item())
            word[:, i + 1] = predicted_word_idx.item()

            # End if <EOS> appears
            if vocab.index2word[predicted_word_idx.item()] == "<EOS>":
                break

        return ' '.join([vocab.index2word[idx] for idx in predicted_captions])
    
    # def predict(self, features, max_length, vocab):
    #     n_samples = features.size(0)

    #     word = torch.tensor([vocab.word2index['<SOS>']] + [0] * (max_length - 1)).view(1, -1)
    #     word = word.repeat(n_samples, 1)

    #     padding_mask = torch.Tensor([True] * max_length).view(1, -1)
    #     padding_mask = padding_mask.repeat(n_samples, 1)

    #     predicted_captions = [[] for _ in range(n_samples)]
    #     is_predicted = [False] * n_samples

    #     for i in range(max_length - 1):
    #         # Update the padding masks
    #         padding_mask[:, i] = False

    #         # Get the model prediction for the next word
    #         output = self.forward(features, word, padding_mask)
    #         output = output[torch.arange(n_samples), [i] * n_samples].clone()
    #         predicted_word_idx = output.argmax(dim=-1)

    #         for idx in range(n_samples):
    #             if is_predicted[idx]:
    #                 continue
    #             predicted_captions[idx].append(predicted_word_idx[idx].item())
    #             if predicted_word_idx[idx].item() == 2:
    #                 is_predicted[idx] = True
    #         if np.all(is_predicted):
    #             break

    #         word[torch.arange(n_samples), [i + 1] * n_samples] = predicted_word_idx.view(-1)
    #     return predicted_captions

class TransformerCaptioner(torch.nn.Module):
    def __init__(self, n_tokens, d_model, n_heads, dim_forward, n_layers, encoder_dim, vocab):
        super(TransformerCaptioner, self).__init__()
        self.encoder =  TransformerEncoder(encoder_dim=encoder_dim,
                                           d_model=d_model)
        
        self.decoder = Decoder(n_tokens=n_tokens,
                               d_model=d_model,
                               n_heads=n_heads,
                               dim_forward=dim_forward,
                               n_layers=n_layers)
        self.vocab = vocab

    def forward(self, images, captions, padding_mask):
        features = self.encoder(images)
        output = self.decoder(features, captions, padding_mask)
        return output

    def generate_caption(self, image, max_length=50):
        features = self.encoder(image)
        predicted_caption = self.decoder.predict(features, max_length, self.vocab)
        return predicted_caption