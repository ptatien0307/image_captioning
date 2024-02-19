
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .vocabulary import Vocabulary

class ImageCaptioningDataset(Dataset):
    """Image Captioning dataset"""

    def __init__(self, csv_file, transform, max_length, freq_threshold=5):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform
        self.max_length = max_length

        self.images = self.dataframe['image']
        self.captions = self.dataframe['caption']

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        image_path = self.images[idx]

        image = cv2.imread(f'dataset/Images/{image_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        # Tokenize caption
        caption_tokens = []
        caption_tokens += [self.vocab.word2index["<SOS>"]]
        caption_tokens += self.vocab.numericalize(caption)
        caption_tokens += [self.vocab.word2index["<EOS>"]]

        input_tokens = caption_tokens[:-1].copy() # input
        target_tokens = caption_tokens[1:].copy() # target

        # Padding input tokens
        cap_length = len(input_tokens)
        padding_size = self.max_length - cap_length
        input_tokens += [0] * padding_size
        target_tokens += [0] * padding_size

        input_tokens = torch.tensor(input_tokens) # input
        target_tokens = torch.tensor(target_tokens) # target

        return image, input_tokens, target_tokens