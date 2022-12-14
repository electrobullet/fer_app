import tkinter as tk

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk

from ctx import emotion_recognizer, face_detector
from dnn.EmotionRecognizer import EmotionRecognizer
from dnn.FaceDetector import FaceDetector


class App:
    def __init__(self, face_detector: FaceDetector, emotion_recognizer: EmotionRecognizer, source: int = 0):
        self._cap = cv.VideoCapture(source)
        self._face_detector = face_detector
        self._emotion_recognizer = emotion_recognizer
        self._emotions = emotion_recognizer.emotions
        self._colors = emotion_recognizer.colors

        self._set_user_interface()

    def _set_user_interface(self):
        self._root = tk.Tk()
        self._root.resizable(False, False)
        self._root.title('Face emotion recognition')

        self._image_label = tk.Label()
        self._image_label.grid(row=0, column=0, rowspan=2 * len(self._emotions))

        self._emotion_labels = []
        self._emotion_values = []
        self._emotion_scales = []

        row = 0
        for i in range(len(self._emotions)):
            self._emotion_labels.append(tk.Label())
            self._emotion_values.append(tk.DoubleVar())
            self._emotion_scales.append(tk.Scale(
                orient=tk.HORIZONTAL,
                length=300,
                showvalue=False,
                from_=0.0,
                to=1.0,
                resolution=0.01,
                state=tk.DISABLED,
                bg='#{:02x}{:02x}{:02x}'.format(*self._colors[i]),
                variable=self._emotion_values[i],
            ))

            self._emotion_labels[-1].grid(row=row, column=1)
            self._emotion_scales[-1].grid(row=row + 1, column=1, padx=10)
            row += 2

    def _get_frame_from_cap(self, cap: cv.VideoCapture) -> np.ndarray:
        _, frame = cap.read()
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    def _show_frame_on_label(self, label: tk.Label, array: np.ndarray):
        # Convert a numpy array to tk image
        image = ImageTk.PhotoImage(Image.fromarray(array))
        # Show it using a tk label widget
        label.configure(image=image)
        label.wtf = image  # type: ignore

    def _main_loop(self):
        frame = self._get_frame_from_cap(self._cap)
        display = frame.copy()
        h, w = display.shape[:2]

        face_predictions = self._face_detector.predict(frame)
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
            emotion_predictions = self._emotion_recognizer.predict(face_crop)

            thickness = 2

            if i == 0:
                for j, value in enumerate(emotion_predictions):
                    self._emotion_labels[j]['text'] = f'{self._emotions[j]}: {value:.2f}'
                    self._emotion_values[j].set(value)

                thickness = 3

            cv.rectangle(display, (x_min, y_min), (x_max, y_max),
                         self._colors[np.argmax(emotion_predictions)], thickness)

        self._show_frame_on_label(self._image_label, display)
        self._root.after(17, self._main_loop)

    def run(self):
        self._main_loop()
        self._root.mainloop()


if __name__ == '__main__':
    app = App(face_detector, emotion_recognizer)
    app.run()
