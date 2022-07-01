import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pandas import read_csv
import os

# Cargar datos
data_path = os.path.join(os.getcwd(), "data/blood-pressure.csv")
dataset = read_csv(data_path, sep=',')

X = dataset[['Age']]
y = dataset[['Pressure']]

regr = LinearRegression()
regr.fit(X, y)

# Salidas Argumentadas
plt.xlabel('Age')
plt.ylabel('Blood pressure')

plt.scatter(X, y,  color='black')
plt.plot(X, regr.predict(X), color='blue')

plt.show()
plt.gcf().clear()