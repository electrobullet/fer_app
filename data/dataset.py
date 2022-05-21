import os

import numpy as np
import pandas as pd
from PIL import Image


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
            pd.read_csv(ferplus_path)[['Usage', 'Image name']],
            pd.read_csv(fer_path)['pixels'],
        ],
        axis=1,
    ).dropna()

    for usage, image_name, pixels in dataframe.values:
        image = Image.fromarray(np.fromstring(pixels, np.uint8, 48 * 48, ' ').reshape(48, 48))
        image_path = os.path.join(base_folder, folder_aliases[usage], image_name)
        image.save(image_path, compress_level=0)


if __name__ == '__main__':
    # unpack('data/pd_images', 'data/fer2013.csv', 'data/fer2013new.csv')
