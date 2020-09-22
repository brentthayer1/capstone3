import os
import sys
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from util import base64_to_pil

app = Flask(__name__)

MODEL_PATH = 'models/FINAL_MODEL_3.h5.h5'
model = load_model(MODEL_PATH)

print('Model loaded. Start serving...')

def model_predict(img, model):
    class_dict = {      0 : 'COVID-19',
                        1 : 'Normal',
                        2 : 'Pneumonia'}

    input_arr = img_to_array(img)

    input_arr = input_arr / 255
    input_arr = np.expand_dims(input_arr, axis=0)
    probs = model.predict(input_arr)
    pred_class = np.argmax(probs)
    pred_class = class_dict[pred_class]
    return pred_class

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        pred = model_predict(img, model)
        return jsonify(result=pred)
    return None

if __name__ == '__main__':
    app.run(port=5000, debug=True)