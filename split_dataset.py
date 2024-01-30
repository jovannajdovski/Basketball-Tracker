import os
import shutil
import random

def split_dataset(data_folder, output_folder, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, seed=42):
    random.seed(seed)

    train_folder = os.path.join(output_folder, 'train')
    valid_folder = os.path.join(output_folder, 'valid')
    test_folder = os.path.join(output_folder, 'test')

    for folder in [train_folder, valid_folder, test_folder]:
        os.makedirs(os.path.join(folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'labels'), exist_ok=True)

    image_files = [f for f in os.listdir(os.path.join(data_folder, 'images')) if f.endswith(('.jpg', '.jpeg', '.png'))]

    random.shuffle(image_files)

    num_train = int(train_ratio * len(image_files))
    num_valid = int(valid_ratio * len(image_files))
    num_test = len(image_files) - num_train - num_valid

    train_files = image_files[:num_train]
    valid_files = image_files[num_train:num_train + num_valid]
    test_files = image_files[num_train + num_valid:]

    move_files(data_folder, train_folder, train_files)
    move_files(data_folder, valid_folder, valid_files)
    move_files(data_folder, test_folder, test_files)


def move_files(source_folder, destination_folder, files):
    for file in files:
        image_path = os.path.join(source_folder, 'images', file)
        label_path = os.path.join(source_folder, 'labels', file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))

        shutil.copy(image_path, os.path.join(destination_folder, 'images'))
        shutil.copy(label_path, os.path.join(destination_folder, 'labels'))


if __name__=="__main__":
    data_folder = './all_data'
    output_folder = './data'
    split_dataset(data_folder, output_folder)