from flask import Flask,render_template,request
import numpy as np
import pickle

app=Flask(__name__)

model=pickle.load(open("glass_model.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
    try:
        features=[float(request.form[key])for key in request.form.keys()]
        scaled_features=scaler.transform([features])
        prediction=model.predict(scaled_features)[0]

        return render_template("index.html",prediction_text=f"Predicted Glass Type:{prediction}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"error:{str(e)}")
    
if __name__=="__main__":
    app.run(debug=True)