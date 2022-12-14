from abc import ABC, abstractmethod, abstractproperty
from typing import List, Tuple

import numpy as np


class EmotionRecognizer(ABC):
    COLORS = {
        'anger': (255, 0, 0),
        'contempt': (255, 128, 0),
        'disgust': (0, 255, 0),
        'fear': (255, 0, 255),
        'happiness': (255, 255, 0),
        'neutral': (224, 224, 224),
        'sadness': (0, 0, 255),
        'surprise': (0, 255, 255),
    }

    @abstractproperty
    def emotions(self) -> List[str]:  # type: ignore
        pass

    @abstractproperty
    def colors(self) -> List[Tuple[int, int, int]]:  # type: ignore
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray:
        pass
