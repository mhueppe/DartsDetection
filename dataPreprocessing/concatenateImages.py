# author: Michael Hüppe
# date: 28.03.2024
# project: /concatenateImages.py
import os

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from PIL import Image

if __name__ == '__main__':
    dataPath = r"C:\Users\mhuep\SNAP\DartsDataGenerator\GeneratedData\Highlight_Total"
    save_path = "../../eyetracking/gazeTask/dataAll_easy.npz"


    def readImages(image_path, target_size = (300,300)):
        img = imageio.imread(image_path)
        img = np.asarray(Image.fromarray(img).resize(target_size), dtype=np.uint8)
        return img


    data = {"labels": []}
    allFiles = os.listdir(dataPath)[:10_000]
    for i, fileName in enumerate(allFiles):
        print(f"{i}/{len(allFiles)}")
        key = str(i)
        data[key] = readImages(os.path.join(dataPath, fileName))
        data["labels"].append(np.asarray(fileName.removesuffix(".png").split("_")[-3:], np.float32))
    np.savez(save_path, **data)
