import os
import sys
import tkinter as tk

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk

from dnn.emotion_recognizers import EmotionRecognizer, Model
from dnn.face_detectors import FaceDetectionRetail0044, FaceDetector


class App:
    def __init__(self, face_detector: FaceDetector, emotion_recognizer: EmotionRecognizer, source: int = 0):
        self.__cap = cv.VideoCapture(source)
        self.__face_detector = face_detector
        self.__emotion_recognizer = emotion_recognizer
        self.__emotions = emotion_recognizer.emotions
        self.__colors = emotion_recognizer.colors

        self.__set_user_interface()

    def __set_user_interface(self):
        self.__root = tk.Tk()
        self.__root.resizable(False, False)
        self.__root.title('Face emotion recognition')

        self.__image_label = tk.Label()
        self.__image_label.grid(row=0, column=0, rowspan=2 * len(self.__emotions))

        self.__emotion_labels = []
        self.__emotion_values = []
        self.__emotion_scales = []

        row = 0
        for i in range(len(self.__emotions)):
            self.__emotion_labels.append(tk.Label())
            self.__emotion_values.append(tk.DoubleVar())
            self.__emotion_scales.append(tk.Scale(
                orient=tk.HORIZONTAL,
                length=300,
                showvalue=False,
                from_=0.0,
                to=1.0,
                resolution=0.01,
                state=tk.DISABLED,
                bg='#{:02x}{:02x}{:02x}'.format(*self.__colors[i]),
                variable=self.__emotion_values[i],
            ))

            self.__emotion_labels[-1].grid(row=row, column=1)
            self.__emotion_scales[-1].grid(row=row + 1, column=1, padx=10)
            row += 2

    def __get_frame_from_cap(self, cap: cv.VideoCapture) -> np.ndarray:
        _, frame = cap.read()
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    def __show_frame_on_label(self, label: tk.Label, array: np.ndarray):
        # Convert a numpy array to tk image
        image = ImageTk.PhotoImage(Image.fromarray(array))
        # Show it using a tk label widget
        label.configure(image=image)
        label.what_the_fuck = image

    def __main_loop(self):
        frame = self.__get_frame_from_cap(self.__cap)
        display = frame.copy()
        h, w = display.shape[:2]

        face_predictions = self.__face_detector.predict(frame)
        face_widths = [x_max - x_min for x_min, _, x_max, _ in face_predictions]
        face_predictions = face_predictions[np.argsort(face_widths)]

        for i in range(len(face_predictions)):
            x_min, y_min, x_max, y_max = face_predictions[i]

            face_width = x_max - x_min
            face_height = y_max - y_min
            face_pad = abs(face_width - face_height) // 2

            if face_width < face_height:
                x_min = np.clip(x_min - face_pad, 0, w)
                x_max = np.clip(x_max + face_pad, 0, w)
            else:
                y_min = np.clip(y_min - face_pad, 0, h)
                y_max = np.clip(y_max + face_pad, 0, h)

            face_crop = frame[y_min:y_max, x_min:x_max]
            emotion_predictions = self.__emotion_recognizer.predict(face_crop)

            thickness = 2

            if i == 0:
                for j, value in enumerate(emotion_predictions):
                    self.__emotion_labels[j]['text'] = f'{self.__emotions[j]}: {value:.2f}'
                    self.__emotion_values[j].set(value)

                thickness = 3

            cv.rectangle(display, (x_min, y_min), (x_max, y_max),
                         self.__colors[np.argmax(emotion_predictions)], thickness)

        self.__show_frame_on_label(self.__image_label, display)
        self.__root.after(17, self.__main_loop)

    def run(self):
        self.__main_loop()
        self.__root.mainloop()


if __name__ == '__main__':
    model_path = os.path.join(
        os.path.dirname(sys.argv[0]),
        'models',
        '5_classes_EfficientNetV2B1_96x96_bs_256_weighted.onnx',
    )

    emotion_recognizer = Model(
        model_path=model_path,
        input_size=(96, 96),
        emotions=[
            'anger',
            # 'contempt',
            # 'disgust',
            # 'fear',
            'happiness',
            'neutral',
            'sadness',
            'surprise',
        ]
    )

    app = App(FaceDetectionRetail0044(), emotion_recognizer)
    app.run()
