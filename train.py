import numpy as np
import pandas as pd
from model import CNN
from sklearn.preprocessing import OneHotEncoder
import joblib

# --- Hyperparameters ---
input_length = 187
num_classes = 5
conv_filters = 12        # Increase filter count
kernel_size = 5
pool_size = 2
hidden_units = 64        # Increase FC layer size
learning_rate = 0.005    # Try a lower learning rate
epochs = 20              # Increase number of epochs
batch_size = 32

# --- Load and preprocess training data ---
train_df = pd.read_csv("Data/mitbih_train.csv", header=None)
X_train = train_df.iloc[:, :-1].values.astype(np.float32)
y_train_raw = train_df.iloc[:, -1].values.astype(int)

# Normalize input signals
X_train = (X_train - np.mean(X_train, axis=1, keepdims=True)) / (np.std(X_train, axis=1, keepdims=True) + 1e-6)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False, categories='auto')
y_train = encoder.fit_transform(y_train_raw.reshape(-1, 1))

# --- Initialize model ---
model = CNN(input_length, num_classes, conv_filters, kernel_size, pool_size, hidden_units)

# --- Training loop ---
for epoch in range(epochs):
    permutation = np.random.permutation(len(X_train))
    X_train = X_train[permutation]
    y_train = y_train[permutation]

    total_loss = 0

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        for x, y in zip(X_batch, y_batch):
            y_hat, cache = model.forward(x)
            grads = model.backward(x, y, cache)
            model.update(grads, learning_rate)

    print(f"Epoch {epoch + 1}/{epochs} completed.")

# Save model
joblib.dump(model, "cnn_model1.pkl")
print("Model saved to cnn_model.pkl")

