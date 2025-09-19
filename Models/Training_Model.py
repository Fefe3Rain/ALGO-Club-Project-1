import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
import os

df = pd.read_csv(r"C:\Users\Rain Sidney\OneDrive\ALGO Club\ALGO-Club-Project-1\Dataset for Exercises\exercise_angles.csv")

X = df.drop(columns=["Label", "Side"])

encoder = LabelEncoder()
y = encoder.fit_transform(df["Label"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.3),  # Prevent overfitting
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(len(np.unique(y)), activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

early_stop = EarlyStopping(
    monitor="val_loss", 
    patience=10, 
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

history = model.fit(
    X_train, y_train,
    epochs=200,  # you can increase to 50+ if needed
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, lr_scheduler]
)

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

# 9️⃣ Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=encoder.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# 🔟 Save the model & scaler for future use
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/exercise_model.keras")
joblib.dump(scaler, "saved_model/scaler.pkl")
joblib.dump(encoder, "saved_model/label_encoder.pkl")
print("✅ Model, scaler, and encoder saved successfully!")