import os

import tensorflow as tf
from tensorflow.keras import layers, models

import json
from tensorflow.keras.callbacks import ModelCheckpoint


# Function to decode and preprocess the image
def load_img(img_path, img_width: int, img_height: int):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float
    img = tf.image.resize(img, [img_width, img_height])  # Resize if necessary
    # get file name
    file_name = tf.strings.split(img_path, "\\")[-1]
    file_name = tf.strings.regex_replace(file_name, ".png", "")
    label = tf.strings.split(file_name, '_')[-3:]
    label = tf.strings.to_number(label, out_type=tf.float32)
    label = tf.reduce_sum(label)
    label = tf.one_hot(tf.cast(label, tf.uint8), 180)
    return img, label


def prepareDataset(data_dir: str, img_width: int, img_height: int, batch_size: int, n_images: int = -1):
    # List of image filenames
    img_files = tf.constant([os.path.join(data_dir, f) for f in os.listdir(data_dir)][:n_images])

    # Create a dataset from the list of filenames
    dataset = tf.data.Dataset.from_tensor_slices(img_files)

    dataset = dataset.map(lambda x: load_img(x, img_width, img_height))

    # Shuffle and batch the dataset
    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == '__main__':
    # Data Preparation
    dataPath = r'..\..\DartsDataGenerator\GeneratedData\Highlight_Total'
    image_height, image_width = (300, 300)
    batch_size = 32
    num_epochs = 1000
    num_classes = len(os.listdir(dataPath))

    dataset = prepareDataset(dataPath, image_height, image_width, batch_size)
    # Define the CNN Model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
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

    history = model.fit(dataset,
                        epochs=num_epochs,
                        callbacks=[checkpoint_callback])

    json.dump(history.history, open(os.path.join(save_path, "history.json"), "w"))
