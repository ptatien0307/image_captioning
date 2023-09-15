from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from keras.models import load_model
import cv2
import os
import sys
from model import Captioner, DecoderLayer, ImageFeatureExtractionLayer, TextFeatureExtractionLayer

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

@keras.saving.register_keras_serializable(package="ImageFeatureExtractionLayer")
class ImageFeatureExtractionLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__()

        pretrained_model = InceptionV3()
        feature_extraction = Model(pretrained_model.input, pretrained_model.layers[-2].output)
        feature_extraction.trainable = False

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
        config.update({'image_fe_model': self.image_fe_model,})
        return config
    

@keras.saving.register_keras_serializable(package="TextFeatureExtractionLayer")
class TextFeatureExtractionLayer(tf.keras.layers.Layer):
    def __init__(self, VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH, name=None, **kwargs):
        super().__init__()

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
    

@keras.saving.register_keras_serializable(package="DecoderLayer")
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH, name=None, **kwargs):
        super().__init__()
        self.image_fe = ImageFeatureExtractionLayer()
        self.text_fe = TextFeatureExtractionLayer(VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH)

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
                'model': self.model,}
        )
        return config


@keras.saving.register_keras_serializable(package="Captioner")
class Captioner(tf.keras.Model):
    def __init__(self, w2i, i2w,
                 MAX_LENGTH,
                 VOCAB_SIZE,
                 EMBEDDING_DIM, name=None, **kwargs):
        super().__init__()
        self.w2i = w2i
        self.i2w = i2w
        self.MAX_LENGTH = MAX_LENGTH
        self.VOCAB_SIZE = VOCAB_SIZE
        self.EMBEDDING_DIM = EMBEDDING_DIM

        self.decoder = DecoderLayer(VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH)

    def call(self, input):
        image, sequence = input
        pred = self.decoder([image, sequence])
        return pred

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'w2i': self.w2i,
            'i2w': self.i2w,
            'MAX_LENGTH': self.MAX_LENGTH,
            'VOCAB_SIZE': self.VOCAB_SIZE,
            'EMBEDDING_DIM': self.EMBEDDING_DIM,
            'decoder': self.decoder,
        })
        return config

    def generate_caption(self, image):
        image = cv2.resize(image, (299, 299))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)


        in_text = 'START_TOKEN'
        for i in range(self.MAX_LENGTH):
            sequence = [self.w2i[w] for w in in_text.split() if w in self.w2i]
            sequence = pad_sequences([sequence], maxlen=self.MAX_LENGTH)
            yhat = self([image, sequence])
            yhat = np.argmax(yhat)
            word = self.i2w[str(yhat)]
            in_text += ' ' + word
            if word == 'END_TOKEN':
                break
        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)

        return final



# weights = model.get_weights(); new_model.set_weights(weights)

# reconstructed_model = load_model('models/model-v1.keras')



# IMAGEDIR = "images/"

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")
# app.mount("/images", StaticFiles(directory="images"), name="images")


# @app.get('/', response_class=HTMLResponse)
# def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})


# @app.post("/upload-files")
# async def create_upload_files(request: Request, file: UploadFile = File(...)):
#     contents = await file.read()

#     # save the file
#     with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
#         f.write(contents)
#     image = cv2.imread(os.path.join('images/', file.filename))
    
#     caption = reconstructed_model(image)

#     show = file.filename
#     return templates.TemplateResponse("index.html", {"request": request, "show": show, "caption": caption})