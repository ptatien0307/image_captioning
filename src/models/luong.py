import torch
from torch.nn import Module, Linear, LSTMCell, Dropout, Embedding
from .modules.attention import Attention
from .modules.base import AttentionEncoder
# from .modules import AttentionEncoder, Attention

class Decoder(Module):
    def __init__(self, vocab_size, embed_dim, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.drop = Dropout(drop_prob)

        self.embedding = Embedding(vocab_size, embed_dim)# Embedding layer
        self.lstm = LSTMCell(input_size=embed_dim + encoder_dim,hidden_size=decoder_dim, bias=True)# LSTM layer
        self.fcn = Linear(decoder_dim + encoder_dim, self.vocab_size)# Linear layer

        # Attention layer
        self.init_h = Linear(encoder_dim, decoder_dim)
        self.init_c = Linear(encoder_dim, decoder_dim)
        self.attention = Attention(attention_dim, encoder_dim, decoder_dim)

    def init_hidden_state(self, features):
        mean_features = features.mean(dim=1)
        h = self.init_h(mean_features)
        c = self.init_c(mean_features)
        return h, c

    def forward_step(self, embed_word, features, hidden_state, cell_state):
        # Compute context and lstm input
        context, attn_weight = self.attention(features, hidden_state)
        lstm_input = torch.cat((embed_word, context), dim=1)

        # Init hidden state
        hidden_state, cell_state = self.lstm(lstm_input, (hidden_state, cell_state))
        
        # Output
        input_linear = torch.cat((hidden_state, context), dim=1)
        output = self.fcn(self.drop(input_linear))

        return output, hidden_state, cell_state, attn_weight

    def forward(self, features, sequences):
        sequence_length = len(sequences[0]) # Sequence length
        preds = torch.zeros(sequences.shape[0], sequence_length, self.vocab_size) # Prediction store

        # Embedding sequence
        embeds = self.embedding(sequences)
        embeds = embeds.to(torch.float32)

        # Init hidden state
        hidden_state, cell_state = self.init_hidden_state(features)

        # Forward pass
        for idx in range(sequence_length):
            embed_word = embeds[:, idx]
            output, hidden_state, cell_state, _ = self.forward_step(embed_word, features, hidden_state, cell_state)
            preds[:, idx] = output

        return preds

    def predict(self, feature, max_length, vocab):     
        word = torch.tensor(vocab.word2index['<SOS>']).view(1, -1) # Starting input
        embed = self.embedding(word) # Embedding sequence

        predicted_captions = []
        attention = []
        hidden_state, cell_state = self.init_hidden_state(feature)

        for idx in range(max_length):
            embed_word = embeds[:, 0]
            output, hidden_state, cell_state, attn_weight = self.forward_step(embed_word, feature, hidden_state, cell_state)
            attention.append(attn_weight.cpu().detach().numpy())

            # Predict word index
            predicted_word_idx = output.argmax(dim=1)
            predicted_captions.append(predicted_word_idx.item())

            # End if <EOS> appears
            if vocab.index2word[predicted_word_idx.item()] == "<EOS>":
                break

            # Send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))

        # Convert the vocab idx to words and return sentence
        return ' '.join([vocab.index2word[idx] for idx in predicted_captions]), attention
    
    # def predict(self, feature, max_length, vocab=None):
    #     # Starting input
    #     word = torch.full((feature.shape[0], 1), vocab.word2index['<SOS>'])

    #     # Embedding sequence
    #     embeds = self.embedding(word)
    #     predicted_captions = torch.zeros(max_length, feature.shape[0])
    #     hidden_state, cell_state = self.init_hidden_state(feature)

    #     for idx in range(max_length):
    #         embed_word = embeds[:, 0]
    #         output, hidden_state, cell_state, attn_weight = self.forward_step(embed_word, feature, hidden_state, cell_state)
    #         # Predict word index
    #         predicted_word_idx = output.argmax(dim=1)
    #         predicted_captions[idx, :] = predicted_word_idx.unsqueeze(0)[:, :]

    #         # Send generated word as the next caption
    #         embeds = self.embedding(predicted_word_idx.unsqueeze(1))
    #     predicted_captions = predicted_captions.permute(1, 0)
    #     return predicted_captions

class LuongCaptioner(Module):
    def __init__(self, vocab_size, embed_dim, attention_dim, encoder_dim, decoder_dim, vocab):
        super(LuongCaptioner, self).__init__()
        self.image_encoder =  AttentionEncoder()
        self.text_decoder = Decoder(vocab_size, embed_dim, attention_dim,
                                                 encoder_dim, decoder_dim)
        self.vocab = vocab

    def forward(self, images, captions):

        features = self.image_encoder(images)
        output = self.text_decoder(features, captions)

        return output

    def generate_caption(self, image, max_length=20):
        feature = self.image_encoder(image)
        predicted_caption, attn_weights = self.text_decoder.predict(feature, max_length, self.vocab)

        return predicted_caption
