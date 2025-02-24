from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open("logreg.pkl", "rb") as f:
    model = pickle.load(f)

# Home Page - HTML Form
@app.route('/')
def home():
    return render_template('index.html')

# Predict from Form Input
@app.route('/predict', methods=["POST"])
def predict_class():
    try:
        age = int(request.form.get("age"))
        new_user = int(request.form.get("new_user"))
        total_pages_visited = int(request.form.get("total_pages_visited"))
        
        prediction = model.predict([[age, new_user, total_pages_visited]])
        return render_template('index.html', result=f"Model Prediction: {prediction[0]}")

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

# Predict from Uploaded File
@app.route('/predict_file', methods=["POST"])
def prediction_test_file():
    try:
        df_test = pd.read_csv(request.files.get("file"))
        prediction = model.predict(df_test)
        return render_template('index.html', file_result=f"File Predictions: {list(prediction)}")

    except Exception as e:
        return render_template('index.html', file_error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
