import sys
import numpy as np
import pyqtgraph as pg
import tensorflow as tf
import os

from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QVBoxLayout, QWidget, QPushButton, QSpinBox, \
    QHBoxLayout, QFrame, QLabel


class ImageWidget(QWidget):
    def __init__(self):
        super().__init__()
        file_path = "../../eyetracking/gazeTask/dataAll_easy.npz"
        self._data = np.load(file_path)
        self._model = tf.keras.models.load_model(r'results\regression_model.h5')

        self._dataPath = r'C:\Users\mhuep\SNAP\DartsDataGenerator\GeneratedData\Highlight_Total'
        self._files = os.listdir(self._dataPath)

        self.image_index = 0
        self.image_multiplier = 1

        # Create pyqtgraph widget
        # Create buttons
        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")

        # Create spinbox
        self.spin_box = QSpinBox()
        self.spin_box.setRange(0, 1000)
        self.spin_box.setValue(1)

        # Layout
        layout = QVBoxLayout(self)

        imageWidget = pg.GraphicsLayoutWidget()
        vb = imageWidget.addViewBox()
        self.x_position = 0
        self._images = {}
        for name, img_data in zip(["File", "Arr", "Difference"], self.getImage(0)):
            img = pg.ImageItem()
            img.setImage(img_data)
            vb.addItem(img)
            # Set the position of the image
            img.setPos(self.x_position, 0)
            self._images[name] = img
            # Increment y position for the next image
            self.x_position += img_data.shape[0] + 10
        layout.addWidget(imageWidget)

        imageWidget_pred = pg.GraphicsLayoutWidget()
        imageWidget_pred.setBackground("w")
        label_img, prediction_img = self.getLabelAndPrediction(self.image_index)

        vb_pred = imageWidget_pred.addViewBox()
        x_position = 0
        self.image_label = pg.ImageItem()
        self.image_label.setImage(label_img)
        vb_pred.addItem(self.image_label)
        # Set the position of the image
        self.image_label.setPos(x_position, 0)
        # Increment y position for the next image
        x_position += prediction_img.shape[0] + 10
        self.image_prediction = pg.ImageItem()
        self.image_prediction.setImage(prediction_img)
        vb_pred.addItem(self.image_prediction)
        # Set the position of the image
        self.image_prediction.setPos(x_position, 0)
        # Increment y position for the next image
        x_position += prediction_img.shape[0] + 10
        layout.addWidget(imageWidget_pred)

        self.label_target = QLabel("")
        self.label_prediction = QLabel("")
        frame_label = QFrame()
        frame_label.setLayout(QHBoxLayout())
        frame_label.layout().addWidget(self.label_target)
        frame_label.layout().addWidget(self.label_prediction)
        layout.addWidget(frame_label)

        frame = QFrame()
        frame.setLayout(QHBoxLayout())
        frame.layout().addWidget(self.previous_button)
        frame.layout().addWidget(self.next_button)
        frame.layout().addWidget(self.spin_box)
        layout.addWidget(frame)
        self.setLayout(layout)
        # Connect signals and slots
        self.previous_button.clicked.connect(self.previous_image)
        self.next_button.clicked.connect(self.next_image)
        self.spin_box.valueChanged.connect(self.change_multiplier)
        self.update_image()

    def getLabelAndPrediction(self, i):
        x = self.getNumpyImg(i)
        label = self.getLabel(i)
        return np.reshape(tf.one_hot(int(label), 180), (18, 10)), np.reshape(self._model(np.expand_dims(x, 0)), (18,10))

    def getNumpyImg(self, i):
        return self._data[str(i)] / 255.

    def getFileImg(self, i):
        fileName = self._files[i]
        img_path = os.path.join(self._dataPath, fileName)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float
        img = tf.image.resize(img, [300, 300])  # Resize if necessary
        return img

    def getImage(self, i):
        img = self.getFileImg(i)
        arr_img = self.getNumpyImg(i)
        diff = np.abs(img - arr_img)
        images = img.numpy(), np.clip((diff / np.max(diff)) * 255 * self.image_multiplier, 0, 255), arr_img
        return images

    def getLabel(self, i):
        fileName = self._files[i]
        file_name = tf.strings.split(fileName, "\\")[-1]
        file_name = tf.strings.regex_replace(file_name, ".png", "")
        label = tf.strings.split(file_name, '_')[-3:]
        label = tf.strings.to_number(label, out_type=tf.float32)
        label = tf.reduce_sum(label)
        return label

    def update_image(self):
        new_images = self.getImage(self.image_index)
        for new_image, (names, image) in zip(new_images, self._images.items()):
            image.setImage(new_image * self.image_multiplier)
        label, prediction = self.getLabelAndPrediction(self.image_index)
        self.image_label.setImage(np.clip(label*100, 0, 255))
        self.image_prediction.setImage(np.clip(prediction*100, 0, 255))
        self.label_target.setText(f"Label: {np.argmax(label)}")
        self.label_prediction.setText(f"Prediction: {np.argmax(prediction)}")
        self.update()

    def previous_image(self):
        self.image_index -= 1
        if self.image_index < 0:
            self.image_index = 0
        self.update_image()

    def next_image(self):
        self.image_index = (self.image_index + 1) % len(self._files)
        self.update_image()

    def change_multiplier(self, value):
        self.image_multiplier = value
        self.update_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = ImageWidget()
    widget.show()
    sys.exit(app.exec_())
