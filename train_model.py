import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pickle



df = pd.read_csv("insurance.csv")
df = pd.get_dummies(df, drop_first=True)

y = np.log(df["charges"])
X = df.drop("charges", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

with open("lr_model.pkl", "wb") as f:
    pickle.dump(model, f)

y_pred = model.predict(X_test)
print("accuracy:", r2_score(y_test, y_pred))
print("loss:", mean_squared_error(y_test, y_pred))


plt.plot([y_test.min(), y_test.max()],[y_pred.min(), y_pred.max()],'r')
plt.xlim(6.8, 11)
plt.ylim(6.8, 11)
plt.scatter(y_test, y_pred)
plt.xlabel("Real Charges")
plt.ylabel("Predicted")

plt.show()






# X = условия задачи, входные данные
# y = правильный ответ
# X_train = задачи для тренировки
# y_train = ответы  для X_train
# X_test = новая контрольная для проверки знаний
# y_test = правильные ответы для X_test
# y_pred = predicted answers for X_test