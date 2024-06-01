import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("D:\\Zunaira\\python\\Models\\diabetes.csv")
print(df.head())

print(df.columns)

x = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
       'DiabetesPedigreeFunction', 'Age']]
y = df[['Outcome']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(len(x_test))

clf = RandomForestClassifier(n_estimators=100, criterion='gini')
clf.fit(x_train, y_train)
print(clf.predict(x_test))
print(x_test)

print(clf.score(x_test, y_test))
