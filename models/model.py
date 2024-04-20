# author: Michael Hüppe
# date: 26.03.2024
# project: /model.py
from tensorflow.keras import models, layers
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    image_height, image_width = (100, 100)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(3, activation='linear')  # Output layer with linear activation for regression
    ])

    # Compile the Model
    model.compile(optimizer='adam',
                  loss=tf.keras.metrics.mean_squared_error,  # Since it's a multiclass classification problem
                  metrics=['mse'])
    model.fit(x=np.random.sample((1, image_height, image_width, 3)), y=np.asarray([[100, 20, 50]]), epochs=1000)
    input_sample = model(np.random.sample((1, image_height, image_width, 3)))
    print(model(np.random.sample((1, image_height, image_width, 3))))
