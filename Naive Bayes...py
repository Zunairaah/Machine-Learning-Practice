import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("D:\\Zunaira\\python\\Models\\diabetes.csv")

print(df.columns)

x = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
       'DiabetesPedigreeFunction', 'Age']]
y = df[['Outcome']]

print(len(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)
print(len(x_test))

model = GaussianNB()
model.fit(x_train, y_train)

print(model.predict(x_test))
print(x_test)

print(model.score(x_test, y_test))