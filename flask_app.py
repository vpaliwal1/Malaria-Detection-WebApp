import tensorflow

import sys
import os
import glob
import re
import numpy as np

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model_vgg19.h5'

# Load your trained model
model = load_model(MODEL_PATH)


def get_labels(code):
    labels = {0: 'Parasitized', 1: 'Uninfected'}

    return labels[code]

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)

    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = get_labels(np.argmax(model.predict(x)[0]))
    print(preds)


    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        f = request.files['file']

        basepath = os.path.dirname(__file__)
        if not os.path.exists('uploads'):
            os.mkdir('uploads')
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=8000, debug=True)