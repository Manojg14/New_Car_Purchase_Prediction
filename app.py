# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model (you need to train & save this)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        salary = float(request.form['salary'])

        features = np.array([[age, salary]])
        prediction = model.predict(features)[0]

        result = "Will Purchase" if prediction == 1 else "Will Not Purchase"

        return render_template('index.html', prediction_text=f'Result: {result}')
    except:
        return render_template('index.html', prediction_text='Error in input')

if __name__ == '__main__':
    app.run(debug=True)