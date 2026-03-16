import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

with open("../lr_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            
            age = float(request.form["age"])
            bmi = float(request.form["bmi"])
            children = int(request.form["children"])
        except ValueError:
            return render_template("index.html", error="Please enter valid numbers for Age, BMI, and Children.")

        sex = request.form["sex"].lower()
        smoker = request.form["smoker"].lower()
        region = request.form["region"].lower()

        
        data = {
            "age": [age],
            "bmi": [bmi],
            "children": [children],
            "sex": [sex],
            "smoker": [smoker],
            "region": [region]
        }
        df = pd.DataFrame(data)
        df = pd.get_dummies(df, drop_first=True)
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)



        log_pred = model.predict(df)

        prediction = np.exp(log_pred)[0]
        prediction = round(prediction, 2)
        
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)