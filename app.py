import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from pyspark import SparkFiles
import flask
import pandas as pd
import tensorflow
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
from PIL import Image
import io
import h5py
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# import keras
from tensorflow.keras.models import load_model 
tensorflow.keras.backend.clear_session() 
app = Flask(__name__)

from poetry import predictor


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    image=request.files["file"].read()
    image = Image.open(io.BytesIO(image))
    
    seed_text=predictor(image)

    return render_template('index.html', prediction_text='Prediction : {}'.format(seed_text))



if __name__ == "__main__":
    app.run(debug=True)
