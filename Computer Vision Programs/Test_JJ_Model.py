import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow import keras
import os

# =============================
# Load model, scaler, encoder
# =============================
model_path = r"C:\Users\Rain Sidney\OneDrive\ALGO Club\ALGO-Club-Project-1\Models\jumpingjack_stage_model.keras"
scaler_path = r"C:\Users\Rain Sidney\OneDrive\ALGO Club\ALGO-Club-Project-1\Models\scaler.pkl"
encoder_path = r"C:\Users\Rain Sidney\OneDrive\ALGO Club\ALGO-Club-Project-1\Models\label_encoder.pkl"

assert os.path.exists(model_path), "Model file not found!"
assert os.path.exists(scaler_path), "Scaler file not found!"
assert os.path.exists(encoder_path), "Encoder file not found!"

model = keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)

# =============================
# Helper Functions
# =============================
def calculate_angle(a, b, c):
    """Calculate angle (in degrees) between three points a, b, c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_features_from_landmarks(landmarks):
    """Extract 10 angles from MediaPipe Pose landmarks in the same order as training."""
    try:
        L = landmarks  # shorthand
        shoulder_angle = calculate_angle([L[11].x, L[11].y], [L[13].x, L[13].y], [L[15].x, L[15].y])
        elbow_angle = calculate_angle([L[13].x, L[13].y], [L[15].x, L[15].y], [L[17].x, L[17].y])
        hip_angle = calculate_angle([L[11].x, L[11].y], [L[23].x, L[23].y], [L[25].x, L[25].y])
        knee_angle = calculate_angle([L[23].x, L[23].y], [L[25].x, L[25].y], [L[27].x, L[27].y])
        ankle_angle = calculate_angle([L[25].x, L[25].y], [L[27].x, L[27].y], [L[29].x, L[29].y])

        # Ground reference angles (approximation)
        shoulder_ground_angle = calculate_angle([L[11].x, 0], [L[11].x, L[11].y], [L[11].x, 1])
        elbow_ground_angle = calculate_angle([L[13].x, 0], [L[13].x, L[13].y], [L[13].x, 1])
        hip_ground_angle = calculate_angle([L[23].x, 0], [L[23].x, L[23].y], [L[23].x, 1])
        knee_ground_angle = calculate_angle([L[25].x, 0], [L[25].x, L[25].y], [L[25].x, 1])
        ankle_ground_angle = calculate_angle([L[27].x, 0], [L[27].x, L[27].y], [L[27].x, 1])

        return [
            shoulder_angle, elbow_angle, hip_angle, knee_angle, ankle_angle,
            shoulder_ground_angle, elbow_ground_angle, hip_ground_angle, knee_ground_angle, ankle_ground_angle
        ]
    except Exception as e:
        print("Error extracting features:", e)
        return []

# =============================
# Setup MediaPipe Pose
# =============================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# =============================
# Load test video
# =============================

cap = cv2.VideoCapture(0)
print("Video opened:", cap.isOpened())

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No more frames or cannot read video.")
        break

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        features = extract_features_from_landmarks(results.pose_landmarks.landmark)

        if len(features) == 10:
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled, verbose=0)
            stage = encoder.inverse_transform([np.argmax(prediction)])[0]
            cv2.putText(frame, f"Stage: {stage}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Landmarks missing!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Jumping Jack Stage Detection", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):  # press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
