import os

if __name__ == '__main__':
    dataPath = r"C:\Users\mhuep\SNAP\DartsDataGenerator\GeneratedData\Samples"
    files = os.listdir(dataPath)
    for file in files:
        label = file.split("_")[-1][:-4]
        category_path = os.path.join(dataPath, label)
        if not os.path.isdir(category_path):
            os.mkdir(category_path)
        print(f"Moving {file} to {label}")
        os.rename(src=os.path.join(dataPath, file), dst=os.path.join(dataPath, category_path, file))