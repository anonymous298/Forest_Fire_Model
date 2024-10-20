from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

with open('pickle_files/ridgescaler.pkl', 'rb') as scalerfile:
    scaler = pickle.load(scalerfile)

with open('pickle_files/ridge.pkl', 'rb') as modelfile:
    model = pickle.load(modelfile)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        temperature = float(request.form.get('temperature'))
        rh = float(request.form.get('rh'))
        ws = float(request.form.get('ws'))
        rain = float(request.form.get('rain'))
        ffmc = float(request.form.get('ffmc'))
        dmc = float(request.form.get('dmc'))
        isi = float(request.form.get('isi'))
        region = float(request.form.get('region'))

        querie = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, region]])
        input_querie = scaler.transform(querie)

        prediction = model.predict(input_querie)

        return render_template('form.html', prediction=round(prediction[0],3))
    
    return redirect(url_for('form'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')