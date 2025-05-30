
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from neural_network import NeuralNetwork

# Load and preprocess data
data = pd.read_csv("diabetes.csv")

# Replace zeros with median for specific columns
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_replace:
    data[col] = data[col].replace(0, data[col].median())

X = data.drop("Outcome", axis=1).values
y = data["Outcome"].values.reshape(-1, 1)

# Standardize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define and train the model
model = NeuralNetwork([X.shape[1], 16, 8, 1], learning_rate=0.01)
losses = model.train(X_train, y_train, epochs=100, batch_size=32)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
plt.title("Confusion Matrix")
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("figures/confusion_matrix.png")
plt.close()

# Loss curve
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("figures/loss_curve.png")
plt.close()
