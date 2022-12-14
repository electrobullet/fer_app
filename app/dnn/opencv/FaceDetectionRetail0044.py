import cv2 as cv
import numpy as np

from ..FaceDetector import FaceDetector


class FaceDetectionRetail0044(FaceDetector):
    def __init__(self, model_path: str, prototxt_path: str) -> None:
        self._model = cv.dnn.readNet(model_path, prototxt_path)

    def predict(self, image: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        result = []

        h, w = image.shape[:2]

        self._model.setInput(cv.dnn.blobFromImage(image))
        predictions = self._model.forward().reshape(-1, 7)
        predictions = predictions[predictions[:, 2] > threshold]

        for _, _, _, x_min, y_min, x_max, y_max in predictions:
            result.append([
                np.clip(x_min * w, 0, w),
                np.clip(y_min * h, 0, h),
                np.clip(x_max * w, 0, w),
                np.clip(y_max * h, 0, h),
            ])

        return np.array(result, dtype=np.int16)
