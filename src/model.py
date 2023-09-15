import tensorflow as tf
from tensorflow import keras

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Dropout, add
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3

from keras import Input
from keras import optimizers
from keras.callbacks import ModelCheckpoint


class ImageFeatureExtractionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        pretrained_model = InceptionV3()
        feature_extraction = Model(pretrained_model.input, pretrained_model.layers[-2].output)
        feature_extraction.trainable = False

        self.image_fe_model = tf.keras.Sequential([
            Input(shape=(299, 299, 3)),
            feature_extraction,
            Dropout(0.5),
            Dense(256, activation='relu')
        ])

    def call(self, input):
        return self.image_fe_model(input)
    

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
    

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH):
        super().__init__()
        self.image_fe = ImageFeatureExtractionLayer()
        self.text_fe = TextFeatureExtractionLayer(VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH)

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


