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
import string
import re


@keras.saving.register_keras_serializable()
def standardize(s):
    s = tf.strings.lower(s)
    s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
    s = tf.strings.join(['[START]', s, '[END]'], separator=' ')
    return s


tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=2000,
    standardize=standardize,
    ragged=True)


@keras.saving.register_keras_serializable()
class ImageFeatureExtractionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        pretrained_model = InceptionV3()
        feature_extraction = Model(
            pretrained_model.input, pretrained_model.layers[-2].output)
        feature_extraction.trainable = False

        self.image_fe_model = tf.keras.Sequential([
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

        self.text_fe_model = tf.keras.Sequential([
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
        self.image_fe = ImageFeatureExtractionLayer()
        self.text_fe = TextFeatureExtractionLayer(
            VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH)

        self.model = tf.keras.Sequential([
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
    def __init__(self,
                 MAX_LENGTH,
                 EMBEDDING_DIM, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer

        self.word_to_index = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary())

        self.index_to_word = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary(),
            invert=True)

        self.MAX_LENGTH = MAX_LENGTH
        self.EMBEDDING_DIM = EMBEDDING_DIM

        self.decoder = DecoderLayer(
            tokenizer.vocabulary_size(), EMBEDDING_DIM, MAX_LENGTH)

    def call(self, input):
        image, sequence = input
        pred = self.decoder([image, sequence])
        return pred

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'tokenizer': self.tokenizer,
            'index_to_word': self.index_to_word,
            'word_to_index': self.word_to_index,
            'MAX_LENGTH': self.MAX_LENGTH,
            'EMBEDDING_DIM': self.EMBEDDING_DIM,
            'decoder': self.decoder,
        })
        return config

    def generate_caption(self, image):
        image = cv2.resize(image, (299, 299))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        y_pred = []
        in_text = '[START]'
        for i in range(self.MAX_LENGTH):
            sequence = tokenizer(in_text)
            sequence = pad_sequences([sequence], maxlen=self.MAX_LENGTH)
            yhat = self([image, sequence])
            yhat = np.argmax(yhat)
            if self.word_to_index('[END]') == tf.constant(yhat, dtype=tf.int64):
                break

            y_pred.append(yhat)

        words = self.index_to_word(y_pred[1:])
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        return result.numpy().decode()
