import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("D:\\Zunaira\\python\\Models\\Iris_dataset.csv")

print(df.columns)

x = df[['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df[['Species']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train.values.ravel())
y_predict = knn.predict(x_test)
print(y_predict)
print(x_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))
print("Precision: ", metrics.precision_score(y_test, y_predict, average='weighted'))
print("Recall: ", metrics.recall_score(y_test, y_predict, average='weighted'))


