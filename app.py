from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_path = 'log_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = request.form.to_dict()
    # Convert data to the format required by your model
    # For example:
    features = [float(data['feature1']), float(data['feature2'])]
    prediction = model.predict([features])[0] 
    
    # Dummy response for example purposes
    print(data)
    # prediction = "Your prediction will be shown here."

    return render_template('./index.html', prediction_text=f'Prediction: ₹{round(prediction)}')

if __name__ == '__main__':
    app.run(debug=True)

