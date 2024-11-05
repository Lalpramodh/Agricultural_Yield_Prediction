import numpy as np
from flask import Flask, request, render_template, redirect, url_for
import pickle

flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def home():
    return render_template("Home.html")  

@flask_app.route("/predict_page")
def predict_page():
    return render_template("index.html")  

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]  
    features = [np.array(float_features)]  
    prediction = model.predict(features)  
    
    return render_template("result.html", prediction_text=f"The Predicted Crop is {prediction[0]}")  

if __name__ == "__main__":
    flask_app.run(debug=True)
