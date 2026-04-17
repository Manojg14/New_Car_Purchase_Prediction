# ---------------- MODEL TRAINING (run separately) ----------------


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
df = pd.read_csv('car_data.csv')

X = df[['Age', 'AnnualSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model =  DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

