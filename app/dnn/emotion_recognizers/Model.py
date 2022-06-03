from typing import List, Tuple

import cv2 as cv
import numpy as np

from .EmotionRecognizer import EmotionRecognizer


class Model(EmotionRecognizer):
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (48, 48), emotions: List[str] = None) -> None:
        self.__model = cv.dnn.readNet(model_path)
        self.__input_size = input_size
        self.__emotions = emotions if emotions else sorted(EmotionRecognizer.COLORS.keys())
        self.__colors = [EmotionRecognizer.COLORS[label] for label in self.__emotions]

    @property
    def emotions(self) -> List[str]:
        return self.__emotions

    @property
    def colors(self) -> List[Tuple[int, int, int]]:
        return self.__colors

    def predict(self, face_crop: np.ndarray) -> np.ndarray:
        image = cv.cvtColor(face_crop, cv.COLOR_RGB2GRAY)
        image = cv.resize(image, self.__input_size, interpolation=cv.INTER_LINEAR)

        self.__model.setInput(cv.dnn.blobFromImage(image))
        predictions = self.__model.forward()

        return predictions.reshape(len(self.__emotions))
