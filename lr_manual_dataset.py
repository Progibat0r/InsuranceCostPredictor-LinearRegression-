import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

dataset = {
    "age": [19, 18, 28, 33, 32, 31, 46, 37, 37, 60, 25, 62, 23, 56, 27, 19, 52, 23],
    "smoker": ["yes", "no", "no", "no", "no", "no", "no", "no", "no", "no", "no", "yes", "no", "no", "yes", "no", "no", "no"],
    "region": ["southwest", "southeast", "southeast", "northwest", "northwest", "southeast", "southeast", "northwest", "northeast", "northwest", "northeast", "southeast", "southwest", "southeast", "southeast", "southwest", "northeast", "northeast"],
    "charges": [16884.924, 1725.5523, 4449.462, 21984.47061, 3866.8552, 3756.6216, 8240.5896, 7281.5056, 6406.4107, 28923.13692, 2721.3208, 27808.7251, 1826.843, 11090.7178, 39611.7577, 1837.237, 10797.3362, 2395.17155]
}
df = pd.DataFrame(dataset)
df = pd.get_dummies(df, drop_first=True)


y = np.log(df["charges"])
X = df.drop("charges", axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))


plt.plot([y_test.min(), y_test.max()],[y_pred.min(), y_pred.max()],'r')
plt.scatter(y_test, y_pred)
plt.xlabel("Real Charges")
plt.ylabel("Predicted")

plt.show()
