
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

from .vocabulary import Vocabulary

class ICDatasetTransformer(Dataset):
    """
        Image Captioning dataset for transformer.

        Args:
            csv_file (str): Path to the CSV file.
            transform: A function/transform to apply to the image.
            max_length (int): Maximum length of the caption sequence.
            freq_threshold (int): Frequency threshold for vocabulary building.

        Attributes:
            dataframe (DataFrame): Pandas DataFrame containing image paths and captions.
            transform (callable): A function/transform to apply to the image data.
            max_length (int): Maximum length of the caption sequence.
            images (Series): Series containing image paths.
            captions (Series): Series containing captions.
            vocab (Vocabulary): Vocabulary object containing word-to-index and index-to-word mappings.
    """

    def __init__(self, csv_file, transform, max_length, freq_threshold=5):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform
        self.max_length = max_length

        self.images = self.dataframe['image']
        self.captions = self.dataframe['caption']

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())


    def __len__(self):
        """ 
            Returns the length of the dataset. 
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
            Fetches the image, input tokens, target tokens, and padding mask for the given index.

            Args:
                idx (int): Index of the sample to fetch.

            Returns:
                tuple: A tuple containing the image, input tokens, target tokens, and padding mask.

        """
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


        # Create padding mask
        padding_mask = torch.ones([self.max_length, ])
        padding_mask[:cap_length] = 0.0
        padding_mask = padding_mask.bool()

        input_tokens = torch.tensor(input_tokens) # input
        target_tokens = torch.tensor(target_tokens) # target

        return image, input_tokens, target_tokens, padding_mask

class ICDataset(Dataset):
    """
        Image Captioning dataset.

        Args:
            csv_file (str): Path to the CSV file.
            transform: A function/transform to apply to the image.
            max_length (int): Maximum length of the caption sequence.
            freq_threshold (int): Frequency threshold for vocabulary building.

        Attributes:
            dataframe (DataFrame): Pandas DataFrame containing image paths and captions.
            transform (callable): A function/transform to apply to the image data.
            max_length (int): Maximum length of the caption sequence.
            images (Series): Series containing image paths.
            captions (Series): Series containing captions.
            vocab (Vocabulary): Vocabulary object containing word-to-index and index-to-word mappings.
    """

    def __init__(self, csv_file, transform, max_length, freq_threshold=5):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform
        self.max_length = max_length

        self.images = self.dataframe['image']
        self.captions = self.dataframe['caption']

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())


    def __len__(self):
        """ 
            Returns the length of the dataset. 
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
            Fetches the image, input tokens, target tokens for the given index.

            Args:
                idx (int): Index of the sample to fetch.

            Returns:
                tuple: A tuple containing the image, input tokens, target tokens.

        """
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