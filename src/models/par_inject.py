import torch
from torch.nn import Module, Linear, LSTM, Dropout, Embedding
from .modules.base import NormalEncoder
# from .modules import NormalEncoder
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Decoder(Module):
    def __init__(self, vocab_size, embed_dim, encoder_dim, decoder_dim, num_layers):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.num_layers = num_layers

        self.embedding = Embedding(vocab_size, embed_dim).to(device) # Embedding layer

        # LSTM layer
        self.lstm = LSTM(input_size=embed_dim + encoder_dim,
                                  hidden_size=decoder_dim,
                                  bias=True,
                                  batch_first=True,
                                  num_layers=self.num_layers,
                                  bidirectional=False).to(device)

        # Linear layer
        self.linear1 = Linear(decoder_dim, decoder_dim).to(device)
        self.linear2 = Linear(decoder_dim, vocab_size).to(device)
        self.drop = Dropout(0.3)

    def init_hidden_state(self, features):
        hidden = torch.zeros(self.num_layers, features.size(0), self.decoder_dim).to(device)
        cell = torch.zeros(self.num_layers, features.size(0), self.decoder_dim).to(device)
        return hidden, cell

    def forward_step(self, features, embed_words):
        # Init hidden state
        hidden_state, cell_state = self.init_hidden_state(features)

        # Concat embedding and context vector
        features = features.unsqueeze(1)                             # (batch_size, feature_dim)
        features = features.repeat(1, embed_words.shape[1], 1)       # (batch_size, sequence_length, feature_dim)
        lstm_input = torch.cat((embed_words, features), dim=2)       # (batch_size, sequence_length, feature_dim + embed_dim)

        # Forward pass
        output, (hn, cn) = self.lstm(lstm_input, (hidden_state, cell_state))

        output = self.linear1(output)
        output = self.drop(output)
        output = self.linear2(output)

        return output

    def forward(self, features, sequences):
        # Embedding sequence
        sequence_length = len(sequences[0]) - 1
        sequences = sequences[:, :-1].to(device)
        embed_words = self.embedding(sequences)
        embed_words = embed_words.to(torch.float32)

        output = self.forward_step(features, embed_words)
        return output


    def predict(self, feature, max_length, vocab):
        # Embedding sequence
        word = torch.tensor(vocab.word2index['<SOS>']).view(1, -1).to(device)
        embed_words = self.embedding(word)
        feature = feature.to(device)

        captions = []
        for idx in range(max_length):
            # Predict word index
            output = self.forward_step(feature, embed_words)[:, -1]
            predicted_word_idx = output.argmax(dim=1)
            captions.append(predicted_word_idx.item())

            if vocab.index2word[predicted_word_idx.item()] == "<EOS>":
                break

            # Procedd with the next predicted word
            next_embed_word = self.embedding(predicted_word_idx).unsqueeze(0)
            embed_words = torch.cat((embed_words, next_embed_word), dim=1)

        # Convert the vocab idx to words and return sentence
        return ' '.join([vocab.index2word[idx] for idx in captions])
    

class ParInjectCaptioner(torch.nn.Module):
    def __init__(self, vocab_size,  vocab, embed_dim, encoder_dim, decoder_dim, num_layers):
        super().__init__()
        self.encoder =  NormalEncoder(encoder_dim)
        self.decoder = Decoder(vocab_size, embed_dim, encoder_dim, decoder_dim, num_layers)
        self.vocab = vocab

    def forward(self, images, captions):

        image_fv = self.encoder(images)
        output = self.decoder(image_fv, captions)

        return output

    def generate_caption(self, image, max_length=20):
        feature = self.encoder(image)
        predicted_caption = self.decoder.predict(feature, max_length, self.vocab)

        return predicted_caption