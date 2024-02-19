import spacy
from collections import Counter

spacy_eng = spacy.load("en_core_web_sm")
class Vocabulary:
    """
        Vocabulary class for tokenizing and numericalizing text data.
        This class handles the creation of a vocabulary from a list of sentences,

        Args:
            freq_threshold (int): Frequency threshold for vocabulary building.

        Attributes:
            index2word (dict): A dictionary mapping index to word tokens.
            word2index (dict): A dictionary mapping word tokens to their corresponding indices.
            freq_threshold (int): Frequency threshold for vocabulary building.

    """
    def __init__(self,freq_threshold):
        self.index2word = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.word2index = {v: k for k, v in self.index2word.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        """
            Return the size of vocabulary
        """
        return len(self.index2word)

    @staticmethod
    def tokenize(text):
        """
            Tokenizes the input text into a list of lowercase tokens.

            Args:
                text (str): Input text to tokenize.

            Returns:
                list: A list of lowercase tokens.

        """
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        """
            Builds the vocabulary from a list of sentences.

            Args:
                sentence_list (list): List of sentences to build the vocabulary from.

        """
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.word2index[word] = idx
                    self.index2word[idx] = word
                    idx += 1

    def numericalize(self, text):
        """
            Converts text into a numericalized sequence using the vocabulary.
            For each word in the text, the corresponding index token for that word
            from the vocabulary built is formed into a list.

            Args:
                text (str): Input text to numericalize.

            Returns:
                list: A list of numericalized tokens.

        """
        tokenized_text = self.tokenize(text)
        return [self.word2index[token] if token in self.word2index else self.word2index["<UNK>"] for token in tokenized_text ]
    