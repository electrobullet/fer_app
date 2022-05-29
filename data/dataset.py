import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

LABELS = [
    'neutral',
    'happiness',
    'surprise',
    'sadness',
    'anger',
    'disgust',
    'fear',
    'contempt',
    'unknown',
    'NF',
]


def unpack(base_folder: str, fer_path: str, ferplus_path: str) -> None:
    folder_aliases = {
        'Training': 'train',
        'PublicTest': 'val',
        'PrivateTest': 'test',
    }

    os.mkdir(base_folder)

    for folder in folder_aliases.values():
        os.mkdir(os.path.join(base_folder, folder))

    dataframe = pd.concat(
        [
            pd.read_csv(ferplus_path),
            pd.read_csv(fer_path, usecols=['pixels']),
        ],
        axis=1,
    ).dropna()

    for key, value in folder_aliases.items():
        subset = dataframe[dataframe['Usage'] == key]
        subset = subset.rename(columns={'Image name': 'filename'})
        subset = subset[['filename', *LABELS]]
        subset.to_csv(os.path.join(base_folder, '..', f'{value}.csv'), index=False)

    for usage, image_name, pixels in dataframe[['Usage', 'Image name', 'pixels']].values:
        image = Image.fromarray(np.fromstring(pixels, np.uint8, 48 * 48, ' ').reshape(48, 48))
        image_path = os.path.join(base_folder, folder_aliases[usage], image_name)
        image.save(image_path, compress_level=0)


def majority(file_path: str):
    dataframe = pd.read_csv(file_path)

    emotion_votes = dataframe[[*LABELS]]
    emotion_votes = emotion_votes.replace(1, 0)

    labels = []

    for line in emotion_votes.values:
        if np.max(line) > 0.5 * np.sum(line):
            labels.append(LABELS[np.argmax(line)])
        else:
            labels.append(np.nan)

    dataframe['class'] = labels
    dataframe = dataframe.dropna()
    dataframe = dataframe[(dataframe['class'] != 'unknown') & (dataframe['class'] != 'NF')]
    dataframe[['filename', 'class']].to_csv(f'majority_{os.path.basename(file_path)}', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_unpack = subparsers.add_parser('unpack')
    parser_unpack.add_argument('-d', '--base_folder', type=str, default='data/images')
    parser_unpack.add_argument('-fer', '--fer_path', type=str, default='data/fer2013.csv')
    parser_unpack.add_argument('-ferplus', '--ferplus_path', type=str, default='data/fer2013new.csv')

    parser_majority = subparsers.add_parser('majority')
    parser_majority.add_argument('-f', '--file_path', type=str, required=True)

    args = parser.parse_args()

    if args.mode == 'unpack':
        unpack(args.base_folder, args.fer_path, args.ferplus_path)

    elif args.mode == 'majority':
        majority(args.file_path)
