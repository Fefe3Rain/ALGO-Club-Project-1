import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras

# 1️⃣ Load dataset
df = pd.read_csv(r"C:\Users\Rain Sidney\OneDrive\ALGO Club\ALGO-Club-Project-1\Dataset for Exercises\exercise_angles.csv")

# 2️⃣ Drop non-numeric columns if not needed
# 'ide' is just "left" or "right" (you could encode it if it's useful)
X = df.drop(columns=["Label", "Side"])

# 3️⃣ Encode labels to numbers
encoder = LabelEncoder()
y = encoder.fit_transform(df["Label"])

# 4️⃣ Normalize features (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6️⃣ Build the neural network
model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),  # Prevent overfitting
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(len(np.unique(y)), activation="softmax")
])

# 7️⃣ Compile
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 8️⃣ Train
history = model.fit(
    X_train, y_train,
    epochs=5,  # you can increase to 50+ if needed
    batch_size=64,
    validation_data=(X_test, y_test)
)

# 9️⃣ Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# 🔟 Save the model & scaler for future use
model.save("exercise_model.h5")
import joblib
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "label_encoder.pkl")
