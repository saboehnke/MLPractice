import tensorflow as tf
import numpy as np
import pandas as pd
import os

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers.experimental import preprocessing

class Classifier:

    def __init__(self):
        self.dataset = None
        self.model = None
        self.loss_fnc = SparseCategoricalCrossentropy(from_logits=True)

    def load_dataset(self, dataset_name):
        dataset = None

        if dataset_name == "mnist":
            dataset = tf.keras.datasets.mnist
        elif dataset_name == "fashion_mnist":
            dataset = tf.keras.datasets.fashion_mnist

        (self.trainingX, self.trainingY), (self.testingX, self.testingY) = dataset.load_data()

        self.normalize_dataset()

        self.trainingX = np.reshape(self.trainingX, (self.trainingX.shape[0], 28, 28, 1))
        self.testingX = np.reshape(self.testingX, (self.testingX.shape[0], 28, 28, 1))

    def generate_model(self):
        # Works well for mnist:
        self.model = Sequential([
            Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ])

    def train(self, epochs=5):
        self.model.compile(optimizer='adam', loss=self.loss_fnc, metrics=['accuracy'])
        self.model.fit(self.trainingX, self.trainingY, epochs=epochs)

    def evaluate(self):
        self.model.evaluate(self.testingX, self.testingY, verbose=2)

    def predict(self, x):
        return self.model.predict(x)

    def normalize_dataset(self):
        self.trainingX = self.trainingX.astype('float32') / 255.0
        self.testingX = self.testingX.astype('float32') / 255.0

dataset = 'fashion_mnist'

classifier = Classifier()

# load existing model
# predictionData = None
# classifier.model = load_model(f'{dataset}_model')
# classifier.predict(predictionData)

# train new model
classifier.load_dataset(dataset)
classifier.generate_model()
classifier.train(epochs=5)
classifier.evaluate()
classifier.model.save(f'{dataset}_model')