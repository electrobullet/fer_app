import os
import subprocess
import sys

import tensorflow as tf

model = tf.keras.models.load_model(sys.argv[1])
model.save(sys.argv[1][:-3])

subprocess.run([
    'python',
    '-m',
    'tf2onnx.convert',
    '--saved-model',
    sys.argv[1][:-3],
    '--output',
    os.path.dirname(sys.argv[1]) + '.onnx',
],
    shell=True,
)
