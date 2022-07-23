import os
import sys, cv2 

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

#from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil

import urllib.request

# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')

print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'model.h5'

# Load your own trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img, model):
    img = cv2.imread(img)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    
    img = np.array(img)
    img = img.astype('float32')
    img /= 255
    img = np.reshape(img ,(1,128,128,3))
    # Preprocessing the image
    #x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='tf')
    #preds = img.shape
    preds = model.predict(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save("image.jpg")

        # Make prediction
        preds = model_predict('image.jpg', model)
        
        label = ['Aedes','Culex']
        result = label[np.argmax(preds)]
        
        return jsonify(result=str(result))

    return None
@app.route('/api', methods=['GET', 'POST'])
def api():
    if request.method == 'GET':
        
        imgurl =request.args.get('img')
        urllib.request.urlretrieve(imgurl, "image.jpg")


        # Make prediction
        preds = model_predict('image.jpg', model)
        print(preds)
        label = ['Aedes','Culex']
        result = label[np.argmax(preds)]
        prob = np.max(preds)
        return jsonify(result=str(result),prob = str(prob))

    return None


if __name__ == '__main__':
    app.run(port=5000, threaded=False, debug=True)

    # Serve the app with gevent
    #http_server = WSGIServer(('0.0.0.0', 5000), app)
    #http_server.serve_forever()
