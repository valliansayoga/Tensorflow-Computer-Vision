import shutil
import csv
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def split_data(
    original_path_train,
    path_to_save_train,
    path_to_save_val,
    split_size=0.1,
):
    """To split Train folder into train and eval 9:1"""
    folders = (Path() / original_path_train).glob("*")

    for folder in folders:
        images_paths = list(folder.glob("*.png"))
        x_train, x_val = train_test_split(images_paths, test_size=split_size)
        for x in x_train:
            path_to_save_train_folder = Path() / path_to_save_train / x.parts[-2]

            if not path_to_save_train_folder.exists():
                path_to_save_train_folder.mkdir(parents=True, exist_ok=True)

            shutil.copy(x, path_to_save_train_folder)

        for x in x_val:
            path_to_save_val_folder = Path() / path_to_save_val / x.parts[-2]

            if not path_to_save_val_folder.exists():
                path_to_save_val_folder.mkdir(parents=True, exist_ok=True)

            shutil.copy(x, path_to_save_val_folder)


def order_test_set(path_to_images, path_to_csv, images_destination):
    testset = {}

    try:
        with open(path_to_csv, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")

            for i, row in enumerate(reader):
                if i == 0:
                    continue

                img_name = row[-1].replace("Test/", "")
                label = row[-2]

                img_folder_dest = Path() / images_destination / label
                img_path_source = Path() / path_to_images / img_name

                if not img_folder_dest.exists():
                    img_folder_dest.mkdir(parents=True, exist_ok=True)

                shutil.copy(img_path_source, img_folder_dest)

    except:
        raise


def create_generators(batch_size, train_data_path, val_data_path, test_data_path):
    preprocessor = ImageDataGenerator(
        rescale=1 / 255,
    )
    train_generator = preprocessor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(60, 60),
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size,
    )
    val_generator = preprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(60, 60),
        color_mode="rgb",
        shuffle=False,  # no need to shuffle on test
        batch_size=batch_size,
    )
    test_generator = preprocessor.flow_from_directory(
        test_data_path,
        class_mode="categorical",
        target_size=(60, 60),
        color_mode="rgb",
        shuffle=False,  # no need to shuffle on test
        batch_size=batch_size,
    )
    return train_generator, val_generator, test_generator


if __name__ == "__main__":
    original_path_train = "./data/Train"
    path_to_save_train = "./data/training_data/train"
    path_to_save_val = "./data/training_data/eval"
    path_to_images = "./data/Test"
    images_destination = "./data/testing_data"
    path_to_csv = "./data/Test.csv"

    split_data(
        original_path_train=original_path_train,
        path_to_save_train=path_to_save_train,
        path_to_save_val=path_to_save_val,
    )

    order_test_set(path_to_images, path_to_csv, images_destination)
