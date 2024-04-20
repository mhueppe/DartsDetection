# author: Michael Hüppe
# date: 26.03.2024
# project: /creaetDataset.py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)

    # Directory containing the images
    data_dir = r'C:\Users\mhuep\SNAP\DartsDataGenerator\GeneratedData\Highlight_Total'
    desired_width = 1000
    desired_height = 1000
    batch_size = 1
    # Function to parse the filename and extract the label
    def parse_filename(filename):
        parts = tf.strings.split(filename, '_')
        label = tf.strings.split(parts[-1], '.')[0]
        label = tf.strings.to_number(tf.strings.split(label, 'x'), out_type=tf.float32)
        return label


    # Function to decode and preprocess the image
    def decode_img(img_path):
        print(img_path)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float
        img = tf.image.resize(img, [desired_height, desired_width])  # Resize if necessary
        # get file name
        file_name = tf.strings.split(img_path, "\\")[-1]
        file_name = tf.strings.regex_replace(file_name, ".png", "")
        label = tf.strings.split(file_name, '_')[-3:]
        label = tf.strings.to_number(label, out_type=tf.float32)
        label = tf.reduce_sum(label)
        label = tf.one_hot(tf.cast(label, tf.uint8), 180)
        return img, label


    # List of image filenames
    img_files = tf.constant([os.path.join(data_dir, f) for f in os.listdir(data_dir)])

    # Create a dataset from the list of filenames
    dataset = tf.data.Dataset.from_tensor_slices(img_files)

    dataset = dataset.map(decode_img)

    # Shuffle and batch the dataset
    dataset = dataset.batch(batch_size)

    # Iterate over the dataset
    for images, labels in dataset:
        # Train your model here using images and labels
        print(labels, images)
        for image, label in zip(images, labels):
            plt.imshow(image)
            plt.title(f"{np.argmax(label)}")
            plt.show()
