# author: Michael Hüppe
# date: 28.03.2024
# project: /test.py
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
if __name__ == '__main__':
    file_path = "../../eyetracking/gazeTask/dataAll_easy.npz"
    data = np.load(file_path)
    model = tf.keras.models.load_model(r'results\regression_model.h5')

    dataPath = r'C:\Users\mhuep\SNAP\DartsDataGenerator\GeneratedData\Highlight_Total'
    for i, fileName in enumerate(os.listdir(dataPath)):
        img_path = os.path.join(dataPath, fileName)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float
        img = tf.image.resize(img, [300, 300])  # Resize if necessary
        file_name = tf.strings.split(fileName, "\\")[-1]
        file_name = tf.strings.regex_replace(file_name, ".png", "")
        label = tf.strings.split(file_name, '_')[-3:]
        label = tf.strings.to_number(label, out_type=tf.float32)
        label = tf.reduce_sum(label)
        prediction = np.argmax(model(np.expand_dims(img, 0)))
        fig, (file_ax, arr_ax, diff_ax) = plt.subplots(ncols=3)
        file_ax.imshow(img)
        arr_img = data[str(i)]/255.
        arr_ax.imshow(arr_img)
        diff = np.abs(img-arr_img)
        im = diff_ax.imshow((diff/np.max(diff))*255)
        file_ax.set_title("From file")
        arr_ax.set_title("From array")
        diff_ax.set_title("Difference")
        file_ax.axis("off")
        arr_ax.axis("off")
        diff_ax.axis("off")
        fig.suptitle(f"Label: {int(np.sum(label))}, Prediction: {prediction}")
        plt.show()