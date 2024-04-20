# author: Michael Hüppe
# date: 28.03.2024
# project: gazeTask/trainModel.py
import json

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class CustomDataGenerator:
    def __init__(self, npz_file, ratio=0.8, batch_size=32, shuffle=True, sum_labels=True, one_hot: bool = True):
        self.data = np.load(npz_file)
        n_labels = len(self.data["labels"])
        n_samples = len(self.data.files) - 1
        assert n_samples == n_labels, f"Number of samples and labels have to match. Found {n_labels} labels and {n_samples} Samples"
        self.num_samples = n_labels  # Number of subarrays
        self.ratio = ratio
        self.input_shape = self.data[str(0)].shape
        self.sum_labels = sum_labels
        self.one_hot = one_hot
        self.label_shape = ([1, ] if sum_labels else self.data["labels"][0].shape) if not one_hot else [180, ]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(self.num_samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def generate_datasets(self):
        train_dataset = tf.data.Dataset.from_generator(self._generate_data,
                                                       (tf.float32, tf.float32),
                                                       (tf.TensorShape(self.input_shape),
                                                        tf.TensorShape(self.label_shape)),
                                                       args=(True,)).batch(self.batch_size).prefetch(
            tf.data.experimental.AUTOTUNE)
        valid_dataset = tf.data.Dataset.from_generator(self._generate_data,
                                                       (tf.float32, tf.float32),
                                                       (tf.TensorShape(self.input_shape),
                                                        tf.TensorShape(self.label_shape)),
                                                       args=(False,)).batch(self.batch_size).prefetch(
            tf.data.experimental.AUTOTUNE)
        return train_dataset, valid_dataset

    def _generate_data(self, is_train):
        start_index = 0 if is_train else int(self.num_samples * self.ratio)
        end_index = int(self.num_samples * self.ratio) if is_train else self.num_samples

        for idx in range(start_index, end_index):
            subarray_key = str(self.indices[idx])
            label = self.data['labels'][self.indices[idx]]
            if self.one_hot:
                label = tf.one_hot(tf.cast(tf.reduce_sum(label), tf.uint8), 180)
            else:
                label /= 60
                label = tf.reduce_sum(label) if self.sum_labels else label
            yield self.data[subarray_key], label


if __name__ == '__main__':
    npz_file = 'dataAll_easy.npz'
    generator = CustomDataGenerator(npz_file, ratio=0.95)
    num_epochs = 500
    train_dataset, valid_dataset = generator.generate_datasets()
    modelPath = r'results\regression_model.h5'
    if modelPath:
        model = tf.keras.models.load_model(modelPath)
    else:
        # Define the CNN Model
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(generator.input_shape[0], generator.input_shape[1], 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(180, activation='softmax'),  # Output layer with linear activation for regression
        ])

        # Compile the Model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',  # Since it's a multiclass classification problem
                      metrics=['accuracy'])

    save_path = '../results'
    os.makedirs(save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(save_path, "regression_model.h5"),
                                          monitor='accuracy',
                                          mode='max',
                                          save_best_only=True,
                                          save_weights_only=False,
                                          verbose=1)

    history = model.fit(train_dataset,
                        epochs=num_epochs,
                        callbacks=[checkpoint_callback])

    json.dump(history.history, open(os.path.join(save_path, "history.json"), "w"))
