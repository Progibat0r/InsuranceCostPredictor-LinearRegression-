import pickle
import pandas as pd
import numpy as np
import sys


with open("lr_model.pkl", "rb") as f:
    model = pickle.load(f)

while True:
    try:
        age = float(input("age: "))
        bmi = float(input("BMI: "))
        children = int(input("children: "))
        sex = input("sex (male/female): ")
        smoker = input("smoker (yes/no): ")
        region = input("region (northeast/northwest/southeast/southwest): ")
        break
    except ValueError:
        print("Please enter valid numbers for age, BMI and children.")



data = {
    "age":[age],
    "bmi":[bmi],
    "children":[children],
    "sex":[sex],
    "smoker":[smoker],
    "region":[region]
    
}


df = pd.DataFrame(data)

df = pd.get_dummies(df, drop_first=True)


df = df.reindex(columns=model.feature_names_in_, fill_value=0)
log_pred = model.predict(df)
prediction = np.exp(log_pred)

print("Predicted insurance cost:", prediction[0])