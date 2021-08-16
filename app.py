from flask import Flask, request, render_template
import pandas as pd
import joblib
import os
import jsonify
import json
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

model=joblib.load("maintanance_model.pkl")

@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    f = request.files['file']
    data = pd.read_excel(f)
    data=data.set_index('Primary Mill').transpose()
    prediction=model.predict(data)
    
    

    return render_template('index.html', prediction_text='Maintanance Action to be taken: ⚠️ {}'.format(prediction)) # rendering the predicted result

  

if __name__ == '__main__':
    app.run()