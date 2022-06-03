from abc import ABC, abstractmethod

import numpy as np


class FaceDetector(ABC):
    @abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray:
        pass
