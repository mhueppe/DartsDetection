# author: Michael Hüppe
# date: 28.03.2024
# project: /resizeImages.py

from PIL import Image
import os


def resize_images(input_dir, output_dir, target_size=(400, 400)):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    fileNames = os.listdir(input_dir)
    totalFiles = len(fileNames)
    # Loop through all files in the input directory
    for i, filename in enumerate(fileNames):
        # Check if the file is an image
        print(f"{i}/{totalFiles}")
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Open the image file
            save_path = os.path.join(output_dir, filename)
            if os.path.exists(save_path):
                continue
            with Image.open(os.path.join(input_dir, filename)) as img:
                # Resize the image
                img_resized = img.resize(target_size, Image.ANTIALIAS)

                # Save the resized image to the output directory
                img_resized.save(save_path)

if __name__ == '__main__':
    # Example usage:
    input_directory = r"C:\Users\mhuep\SNAP\DartsDataGenerator\GeneratedData\Highlight_Random_complex_multiple"
    output_directory = r"C:\Users\mhuep\SNAP\DartsDataGenerator\GeneratedData\Highlight_Random_complex_multiple_small"
    resize_images(input_directory, output_directory)
