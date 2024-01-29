import os
import cv2
import random
import numpy as np
from math import cos, sin, radians
from glob import glob

data_root_path = './data/'
output_path = './augmented_data/'
augment_data_factor = 1.0

def rotate_point(x, y, angle, center):
    angle_rad = radians(angle)
    new_x = cos(angle_rad) * (x - center[0]) - sin(angle_rad) * (y - center[1]) + center[0]
    new_y = sin(angle_rad) * (x - center[0]) + cos(angle_rad) * (y - center[1]) + center[1]
    return new_x, new_y

def augment_data(image_path, label_path, output_image_path, output_label_path):
    image = cv2.imread(image_path)

    with open(label_path, 'r') as label_file:
        lines = label_file.readlines()

    if not lines:
        return

    flipx = random.choice([True, False])
    if flipx:
        image = cv2.flip(image, 1)

    angle = random.uniform(-10, 10)

    brightness_factor = random.uniform(0.8, 1.2)
    image = np.clip(image * brightness_factor, 0, 255)

    cv2.imwrite(output_image_path, image)

    with open(output_label_path, 'w') as output_label_file:
        for line in lines:
            label_data = line.strip().split()
            if not label_data:
                continue
            class_name = label_data[0]
            center_x, center_y, width, height = map(float, label_data[1:])

            if flipx:
                center_x = 1 - center_x

            new_center_x, new_center_y = rotate_point(center_x, center_y, angle, (0.5, 0.5))

            output_label_file.write(f'{class_name} {new_center_x} {new_center_y} {width} {height}\n')


if __name__ == '__main__':
    os.makedirs(output_path, exist_ok=True)

    split_folder = 'train'
    image_files = glob(os.path.join(data_root_path, split_folder, 'images', '*.jpg')) + glob(os.path.join(data_root_path, split_folder, 'images', '*.png'))

    num_images = int(len(image_files) * augment_data_factor)
    selected_images = random.sample(image_files, num_images)

    for i, image_path in enumerate(selected_images):
        label_path = os.path.join(data_root_path, split_folder, 'labels', os.path.splitext(os.path.basename(image_path))[0] + '.txt')
        output_image_path = os.path.join(output_path, split_folder, 'images', os.path.basename(image_path))
        output_label_path = os.path.join(output_path, split_folder, 'labels', os.path.basename(label_path))

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_label_path), exist_ok=True)

        if i % (num_images // 10) == 0:
            print(f'Still processing {split_folder} image {i + 1}/{num_images}')
        augment_data(image_path, label_path, output_image_path, output_label_path)
