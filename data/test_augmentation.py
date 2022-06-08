import argparse

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--base_folder', type=str, default='data/images/train')
    parser.add_argument('-f', '--csv_file', type=str, default='data/majority_train.csv')

    args = parser.parse_args()

    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.05,
        shear_range=np.rad2deg(np.arctan(0.05)),  # used to set the value by fraction instead of an angle
        horizontal_flip=True,
        fill_mode='reflect',
    )

    target_size = (512, 512)

    data_flow = data_generator.flow_from_dataframe(
        dataframe=pd.read_csv(args.csv_file),
        directory=args.base_folder,
        target_size=target_size,
        interpolation='bilinear',
        color_mode='grayscale',
        batch_size=1,
    )

while True:
    image_1 = next(data_flow)[0][0].reshape(target_size).astype(np.uint8)
    image_2 = next(data_flow)[0][0].reshape(target_size).astype(np.uint8)

    cv.imshow('image_1', image_1)
    cv.imshow('image_2', image_2)

    if cv.waitKey(0) == ord('q'):
        break
