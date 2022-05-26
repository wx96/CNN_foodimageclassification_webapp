from __future__ import division, print_function
# coding=utf-8
import cv2
import sys
import os
import glob
import re
import numpy as np
from PIL import Image

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename    #Helper object for HTML generation - return secure version ofa filename
from gevent.pywsgi import WSGIServer    #web server gateway interface - standardize interface between webserver and python web frameworks/apps

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving in http://localhost:5000')

def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((200, 200))
    return np.array(image)
def get_food_name(label):
    if label==0:
        return "Fried chicken = RM 3"
    if label==1:
        return "Keli = RM 2"
    if label==2:
        return "Rice = RM 1"
def model_predict(img_path, model):
    ar=convert_to_array(img_path)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    food=get_food_name(label_index)
    return food


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)
        return result

if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
