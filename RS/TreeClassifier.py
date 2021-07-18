import cv2
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, experimental
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class TreeClassifier():
    def __init__(self):
        self.batch_size = 32
        self.img_height = 150
        self.img_width = 150
        self.data_directory = f'{os.getcwd()}\\trees'
        self.model = None
        self.loss_fnc = SparseCategoricalCrossentropy(from_logits=True)
        self.classes = ['oak', 'willow', 'yew']

    def load_dataset(self):
        self.training_set = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_directory,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            # color_mode='grayscale'
        )

        self.testing_set = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_directory,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            # color_mode='grayscale'
        )

        self.training_set = self.training_set.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.testing_set = self.testing_set.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    def create_model(self):
        base_model = tf.keras.applications.ResNet50(input_shape=(self.img_height, self.img_width, 3),
                                               include_top=False,
                                               weights='imagenet')
        base_model.trainable = False
        self.model = Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(self.classes))
        ])

        # self.model = Sequential([
        #     experimental.preprocessing.Rescaling(1.0 / 255),
        #     Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 1)),
        #     MaxPooling2D(2, 2),
        #     Conv2D(64, (3, 3), activation='relu'),
        #     MaxPooling2D(2, 2),
        #     Conv2D(64, (3, 3), activation='relu'),
        #     Flatten(),
        #     Dense(128, activation='relu'),
        #     Dropout(0.2),
        #     Dense(len(self.classes), activation='softmax')
        # ])

    def load_model(self):
        self.model = load_model(f'TreeClassifier_model.h5')

    def train(self, epochs=5):
        self.model.compile(optimizer='adam', loss=self.loss_fnc, metrics=['accuracy'])
        self.model.fit(self.training_set, validation_data=self.testing_set, epochs=epochs)

    def evaluate(self):
        self.model.evaluate(self.testing_set, verbose=2)

    def predict(self, x):
        return self.model.predict(x)

    def predict_from_image(self, img):
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)

        return self.predict(img_batch)

    def predict_from_path(self, path):
        img = cv2.imread(path)
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)

        return self.predict(img_batch)

    def augment_data(self):
        generator = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        for class_folder in os.listdir(self.data_directory):
            for file in os.listdir(f'{self.data_directory}\\{class_folder}'):
                if not file.endswith('.png'):
                    continue

                # if '_' in file:
                #     os.remove(f'{self.data_directory}\\{class_folder}\\{file}')

                curr_img = cv2.imread(f'{self.data_directory}\\{class_folder}\\{file}')
                img = image.img_to_array(curr_img)
                img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))

                i = 0
                for batch in generator.flow(img, save_prefix='aug', save_format='png'):
                    cv2.imwrite(f'{self.data_directory}\\{class_folder}\\{file[:-4]}_{i}.png', image.img_to_array(batch[0]))
                    i += 1
                    if i > 4:
                        break

# training = False
# classifier = TreeClassifier()

# # classifier.augment_data()

# if training:
#     classifier.load_dataset()
#     classifier.create_model()
#     classifier.train(epochs=1)
#     classifier.evaluate()
#     classifier.model.save(f'TreeClassifier_model.h5')
# else:
#     classifier.load_model()

#     for file in os.listdir(os.getcwd()):
#         if not file.endswith('.png'):
#             continue

#         predictions = classifier.predict_from_path(file)
#         prediction_idx = np.argmax(predictions[0])

#         print(f'File: {os.fsdecode(file)}, Prediction: {classifier.classes[prediction_idx]}, Certainty: {predictions[0][prediction_idx]}')