import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load and preprocess test data ---
test_df = pd.read_csv("Data/mitbih_test.csv", header=None)
X_test = test_df.iloc[:, :-1].values.astype(np.float32)
y_test_raw = test_df.iloc[:, -1].values.astype(int)

# Normalize input signals
X_test = (X_test - np.mean(X_test, axis=1, keepdims=True)) / (np.std(X_test, axis=1, keepdims=True) + 1e-6)

# One-hot encode labels for consistency
encoder = OneHotEncoder(sparse_output=False, categories='auto')
y_test = encoder.fit_transform(y_test_raw.reshape(-1, 1))

# --- Load trained model (via joblib) ---

model = joblib.load("cnn_model1.pkl")

# --- Evaluation ---
y_pred_labels = []
y_true_labels = []

for x, y_true in zip(X_test, y_test):
    y_hat, _ = model.forward(x)
    y_pred = np.argmax(y_hat)
    y_true = np.argmax(y_true)
    y_pred_labels.append(y_pred)
    y_true_labels.append(y_true)

# --- Metrics ---
accuracy = accuracy_score(y_true_labels, y_pred_labels)
precision = precision_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
recall = recall_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)

# --- Results ---
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# Graph 1: Performance Metrics Bar Chart
metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}

plt.figure(figsize=(8, 6))
plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red'])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Metrics")
plt.show()

# Graph 2: Confusion Matrix Heatmap
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
