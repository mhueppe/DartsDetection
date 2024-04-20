# author: Michael Hüppe
# date: 27.03.2024
# project: /evaluation.py
import numpy as np
import tensorflow as tf
import os
from models.train import prepareDataset
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def createImg(arr):
    # Calculate the size of the square matrix
    size = int(np.ceil(np.sqrt(len(arr))))

    # Calculate the number of elements to pad
    pad_length = size ** 2 - len(arr)

    # Pad the array with zeros if necessary
    padded_arr = np.pad(arr, (0, pad_length), mode='constant')

    # Reshape the padded array into a square matrix
    square_matrix = padded_arr.reshape((size, size))
    return square_matrix

if __name__ == '__main__':
    model = tf.keras.models.load_model(r'results\regression_model.h5')
    model.summary()

    dataPath = r'C:\Users\mhuep\SNAP\DartsDataGenerator\GeneratedData\Highlight_Total'
    image_height, image_width = (model.layers[0].input.shape[1],model.layers[0].input.shape[2])
    batch_size = 20
    num_epochs = 5
    num_classes = len(os.listdir(dataPath))

    dataset = prepareDataset(dataPath, image_height, image_width, batch_size)

    original_shape = (300, 300)
    desired_height = original_shape[0]
    pad = 0
    desired_width = original_shape[1]+pad
    # Create a figure and subplots
    fig, (image_ax, label_ax, prediction_ax) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 1, 1]})
    fig.suptitle("Total Count Predictor")
    # Initialize empty images
    label_im = label_ax.imshow(np.random.random((18,10)),  animated=True)
    prediction_im = prediction_ax.imshow(np.random.random((18,10)),  animated=True)
    image_im = image_ax.imshow(np.zeros((desired_width, desired_width)),  animated=True)
    image_ax.axis("off")
    label_ax.axis("off")
    prediction_ax.axis("off")

    label_title = label_ax.text(0.5, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, ha="center")
    prediction_title = prediction_ax.text(0.5, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, ha="center")
    difference_title = image_ax.text(10, 20, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5})
    image_ax.set_ylabel("Error")
    image_ax.set_xlabel("Samples")
    plot = image_ax.plot(np.arange(10), color='red')
    differences = []
    # Function to update the images
    def update(frame):
        label, image, prediction = frame
        image_ax.set_title("Input")
        image_im.set_array(image)
        image_ax.set_ylabel("Error")
        image_ax.set_xlabel("Samples")
        l = np.argmax(label)
        p = np.argmax(prediction)
        label_title.set_text(f"{l}")
        diff = np.abs(l-p)
        difference_title.set_text(f"Diff: {diff}")
        differences.append(diff/180*desired_height)
        if len(differences) > 200:
            differences.pop(0)
        plot[0].set_data(np.arange(len(differences)), desired_height-np.asarray(differences))
        label_im.set_array(np.asarray(label).reshape((18, 10)))
        prediction_title.set_text(f"{p}")
        prediction_im.set_array(np.asarray(prediction).reshape((18, 10)))
        return prediction_im, image_im, label_im, label_title, prediction_title, difference_title, plot[0]


    # Function to generate frames
    showBatches = 100
    plt.show()
    def data_generator():
        i = 0
        for images, labels in dataset:
            predictions = model(images)
            i +=1

            for label, image, prediction in zip(labels, images, predictions):
                # Create a new array filled with white pixels
                extended_image = np.ones((desired_width, desired_height, 3))

                # Copy the original image into the new array
                extended_image[:original_shape[1], :, :] = image
                yield label, extended_image, prediction
            if i > showBatches-1:
                return
    # Create the animation
    ani = FuncAnimation(fig, update, frames=data_generator, blit=True, interval=100)

    ani.save(filename="animation.mp4",  dpi=300, progress_callback = lambda i, n: print(f'Saving frame {i+1}/{batch_size*showBatches}'))
