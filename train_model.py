import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("student_scores.csv")

# Prepare input (X) and output (y)
X = data[['Hours']].values
y = data['Score'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
with open("slr_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved as slr_model.pkl")
