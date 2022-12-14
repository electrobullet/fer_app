from abc import ABC, abstractmethod

import numpy as np


class FaceDetector(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        pass
