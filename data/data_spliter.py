from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import os


def data_spliter():
    """ Splits the dataset given as jpg files in class folders, into train and test datasets in the same format"""

    dataset_path = Path("unsplit_dataset")
    split_path = Path("split_dataset")
    train_path = split_path / "train"
    test_path = split_path / "test"

    # If the image folder doesn't exist, download it and preapare it

    if split_path.is_dir():
        print(f"{split_path} directory already exist. Skipping download...")
    else:
        print(f"{split_path} doesn't exist, creating one...")
        split_path.mkdir(parents=True, exist_ok=True)
        train_path.mkdir(parents=True, exist_ok=True)
        test_path.mkdir(parents=True, exist_ok=True)
    for folder in list(os.scandir(dataset_path)):
        image_set = list(os.scandir(folder))
        train_set, test_set = train_test_split(image_set, test_size=0.2, shuffle=False)

        class_path_train = train_path / Path(folder).name
        class_path_train.mkdir(parents=True, exist_ok=True)

        class_path_test = test_path / Path(folder).name
        class_path_test.mkdir(parents=True, exist_ok=True)
        for old_image_path in train_set:
            new_image_path = class_path_train / Path(old_image_path).name
            shutil.copy(old_image_path, new_image_path)

        for old_image_path in test_set:
            new_image_path = class_path_test / Path(old_image_path).name
            shutil.copy(old_image_path, new_image_path)


data_spliter()
