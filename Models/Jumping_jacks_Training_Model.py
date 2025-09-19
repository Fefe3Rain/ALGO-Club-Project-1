import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\Rain Sidney\OneDrive\ALGO Club\ALGO-Club-Project-1\Dataset for Exercises\exercise_angles.csv")

# Automatically add Stage column based on Shoulder_Angle
# Adjust the threshold (e.g., 90) if needed after checking data distribution
df["Stage"] = df["Shoulder_Angle"].apply(lambda x: "open" if x > 90 else "close")

# Drop non-numeric columns not needed for training
# (We keep Stage because it's part of the label)
X = df.drop(columns=["Stage", "Label"])  # Keep all angle columns, drop original exercise label

# Encode Stage column as the label we want to predict
# 0 = close, 1 = open
encoder = LabelEncoder()
y = encoder.fit_transform(df["Stage"])

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Build the neural network
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid")
])

# Compile
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Train with early stopping
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Plot accuracy & loss curves
plt.figure(figsize=(8, 4))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save model, scaler, and encoder
model.save("jumpingjack_stage_model.keras")
import joblib
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "label_encoder.pkl")
print("✅ Model, scaler, and encoder saved successfully!")
