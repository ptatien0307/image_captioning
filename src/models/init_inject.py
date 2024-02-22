import torch
from torch.nn import Module, Linear, LSTM, Dropout, Embedding
from .modules.base import NormalEncoder
# from .modules import NormalEncoder

class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, encoder_dim, decoder_dim, num_layers):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.num_layers = num_layers


        # Embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)

        # LSTM layer
        self.lstm = torch.nn.LSTM(input_size=embed_dim,
                                  hidden_size=decoder_dim,
                                  bias=True,
                                  batch_first=True,
                                  num_layers=self.num_layers,
                                  bidirectional=False)

        # Linear layer
        self.linear1 = torch.nn.Linear(decoder_dim, decoder_dim)
        self.linear2 = torch.nn.Linear(decoder_dim, vocab_size)
        self.drop = torch.nn.Dropout(0.3)

    def init_hidden_state(self, features):
        hidden = features.repeat(self.num_layers, 1, 1)
        cell = features.repeat(self.num_layers, 1, 1)
        return hidden, cell

    def forward_step(self, features, embed_words):
        # Init hidden state
        hidden_state, cell_state = self.init_hidden_state(features)

        # Forward pass
        output, (hn, cn) = self.lstm(embed_words, (hidden_state, cell_state))

        output = self.linear1(output)
        output = self.drop(output)
        output = self.linear2(output)

        return output

    def forward(self, features, sequences):
        # Embedding sequence
        embed_words = self.embedding(sequences)
        embed_words = embed_words.to(torch.float32)

        output = self.forward_step(features, embed_words)
        return output


    def predict(self, feature, max_length, vocab):
        # Embedding sequence
        word = torch.tensor(vocab.word2index['<SOS>']).view(1, -1)
        embed_words = self.embedding(word)

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
    
    # def predict(self, features, max_length, vocab):
    #     # Embedding sequence
    #     words = torch.full((features.shape[0], 1), vocab.word2index['<SOS>'])
    #     embed_words = self.embedding(words)

    #     predicted_captions = torch.zeros(features.shape[0], max_length)

    #     for idx in range(max_length):
    #         # Predict word index
    #         output = self.forward_step(features, embed_words)[:, -1]
    #         predicted_word_idx = output.argmax(dim=1)
    #         predicted_captions[:, idx] = predicted_word_idx.unsqueeze(0)[:, :]

    #         # Procedd with the next predicted word
    #         next_embed_word = self.embedding(predicted_word_idx).unsqueeze(0)
    #         next_embed_word = next_embed_word.permute(1, 0, 2)
    #         embed_words = torch.cat((embed_words, next_embed_word), dim=1)

    #     return predicted_captions

class InitInjectCaptioner(torch.nn.Module):
    def __init__(self, vocab_size,  vocab, embed_dim, encoder_dim, decoder_dim, num_layers):
        super().__init__()
        self.encoder =  NormalEncoder(encoder_dim)
        self.decoder = Decoder(vocab_size, embed_dim, encoder_dim, decoder_dim, num_layers)
        self.vocab = vocab

    def forward(self, images, captions):
        features = self.encoder(images)
        output = self.decoder(features, captions)

        return output

    def generate_caption(self, image, max_length=20):
        feature = self.encoder(image)
        predicted_caption = self.decoder.predict(feature, max_length, self.vocab)

        return predicted_caption