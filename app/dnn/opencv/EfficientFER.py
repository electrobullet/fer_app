from typing import List, Tuple

import cv2 as cv
import numpy as np

from ..EmotionRecognizer import EmotionRecognizer


class EfficientFER(EmotionRecognizer):
    def __init__(self, model_path: str, input_size: Tuple[int, int],
                 emotions: List[str] = None) -> None:  # type: ignore
        self._model = cv.dnn.readNet(model_path)
        self._input_size = input_size
        self._emotions = emotions if emotions else sorted(EmotionRecognizer.COLORS.keys())
        self._colors = [EmotionRecognizer.COLORS[label] for label in self._emotions]

    @property
    def emotions(self) -> List[str]:
        return self._emotions

    @property
    def colors(self) -> List[Tuple[int, int, int]]:
        return self._colors

    def predict(self, face_crop: np.ndarray) -> np.ndarray:
        image = cv.cvtColor(face_crop, cv.COLOR_RGB2GRAY)
        image = cv.resize(image, self._input_size, interpolation=cv.INTER_LINEAR)

        self._model.setInput(cv.dnn.blobFromImage(image))
        predictions = self._model.forward()

        return predictions.reshape(len(self._emotions))
