import os
import sys

from dnn.opencv.EfficientFER import EfficientFER
from dnn.opencv.FaceDetectionRetail0044 import FaceDetectionRetail0044

face_detector = FaceDetectionRetail0044(
    os.path.join(os.path.dirname(sys.argv[0]), 'models', 'face-detection-retail-0044.caffemodel'),
    os.path.join(os.path.dirname(sys.argv[0]), 'models', 'face-detection-retail-0044.prototxt'),
)

emotion_recognizer = EfficientFER(
    os.path.join(os.path.dirname(sys.argv[0]), 'models', '5_classes_EfficientNetV2B1_96x96_bs_256_weighted.onnx'),
    (96, 96),
    emotions=[
        'anger',
        # 'contempt',
        # 'disgust',
        # 'fear',
        'happiness',
        'neutral',
        'sadness',
        'surprise',
    ],
)
