import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv("D:\\Zunaira\\python\\Models\\Iris_dataset.csv")

# Display the column names
print(df.columns)

# Feature and target variables
x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Initialize and train the SVM model
model = SVC(kernel='linear')
model.fit(x_train, y_train)

# Make predictions
predictions = model.predict(x_test)
print(predictions)
print(x_test)

# Print model accuracy
accuracy = model.score(x_test, y_test)
print(f'Model Accuracy: {accuracy:.2f}')

# Add a column for the predictions
x_test = x_test.copy()
x_test['Predicted Species'] = predictions
x_test['Actual Species'] = y_test.values
