import tensorflow as tf
from tensorflow import keras

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Dropout, add
from keras.applications.inception_v3 import InceptionV3

from keras import Input

import cv2
import numpy as np


@keras.saving.register_keras_serializable()
class ImageFeatureExtractionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Load pretrained model
        pretrained_model = InceptionV3()
        feature_extraction = Model(
            pretrained_model.input, pretrained_model.layers[-2].output)
        feature_extraction.trainable = False

        # Build a model to extract feature from image
        self.image_fe_model = Sequential([
            Input(shape=(299, 299, 3)),
            feature_extraction,
            Dropout(0.5),
            Dense(256, activation='relu')
        ])

    def call(self, input):
        return self.image_fe_model(input)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'image_fe_model': self.image_fe_model, })
        return config


@keras.saving.register_keras_serializable()
class TextFeatureExtractionLayer(tf.keras.layers.Layer):
    def __init__(self, VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH):
        super().__init__()
        # Build a model to extract feature from text
        self.text_fe_model = Sequential([
            Input(shape=(MAX_LENGTH,)),
            Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True),
            Dropout(0.5),
            LSTM(256)
        ])

    def call(self, input):
        return self.text_fe_model(input)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'text_fe_model': self.text_fe_model})
        return config


@keras.saving.register_keras_serializable()
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH):
        super().__init__()
        # A decoder includes a image feature extractor and text feature extractor
        self.image_fe = ImageFeatureExtractionLayer()
        self.text_fe = TextFeatureExtractionLayer(
            VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH)

        self.model = Sequential([
            Dense(256, activation='relu'),
            Dense(VOCAB_SIZE, activation='softmax')
        ])

    def call(self, input):
        image, text = input
        image_fe_in = self.image_fe(image)
        text_fe_in = self.text_fe(text)

        merge_fe = add([image_fe_in, text_fe_in])

        out = self.model(merge_fe)

        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({'image_fe': self.image_fe,
                       'text_fe': self.text_fe,
                       'model': self.model, }
                      )
        return config

    def get_config(self):
        config = super().get_config().copy()
        config.update({'image_fe': self.image_fe,
                       'text_fe': self.text_fe,
                       'model': self.model, }
                      )
        return config


@keras.saving.register_keras_serializable()
class Captioner(tf.keras.Model):
    def __init__(self, w2i, i2w,
                 VOCAB_SIZE, MAX_LENGTH, EMBEDDING_DIM, **kwargs):
        super().__init__()
        self.VOCAB_SIZE = VOCAB_SIZE
        self.MAX_LENGTH = MAX_LENGTH
        self.EMBEDDING_DIM = EMBEDDING_DIM

        self.w2i = w2i
        self.i2w = i2w
        self.decoder = DecoderLayer(VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH)

    def call(self, input):
        image, sequence = input
        pred = self.decoder([image, sequence])
        return pred

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'i2w': self.i2w,
            'w2i': self.w2i,
            'decoder': self.decoder,
            'MAX_LENGTH': self.MAX_LENGTH,
            'VOCAB_SIZE': self.VOCAB_SIZE,
            'EMBEDDING_DIM': self.EMBEDDING_DIM,
        })
        return config

    def generate_caption(self, image):
        image = cv2.resize(image, (299, 299))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        in_text = '[START]'
        for i in range(self.MAX_LENGTH):
            # Tokenize input
            sequence = [self.w2i[w] for w in in_text.split() if w in self.w2i]

            # Padding
            sequence = pad_sequences([sequence], maxlen=self.MAX_LENGTH)

            # Predict
            yhat = self([image, sequence])
            yhat = np.argmax(yhat)

            # Convert index into word
            word = self.i2w[str(yhat)]

            # Add predicted word add the end of the input text
            in_text += ' ' + word

            # If word is [END] token, then break
            if word == '[END]':
                break

        final = in_text.split()
        final = final[1: -1]
        final = ' '.join(final)

        return final
