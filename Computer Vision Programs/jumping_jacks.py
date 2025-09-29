import cv2
import mediapipe as mp

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# OpenCV video capture (0 = webcam)
cap = cv2.VideoCapture(r"C:\Users\Rain Sidney\OneDrive\ALGO Club\ALGO-Club-Project-1\Sample Video\Full Body Exercises - Jumping Jacks.mp4")

rep_count = 0
stage = "down"  # can be "up" or "down"

def get_angle(a, b, c):
    """Calculate angle between three points a, b, c"""
    import numpy as np
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = result.pose_landmarks.landmark

        # Extract key points (normalized)
        left_shoulder = [landmarks[11].x * w, landmarks[11].y * h]
        left_elbow = [landmarks[13].x * w, landmarks[13].y * h]
        left_wrist = [landmarks[15].x * w, landmarks[15].y * h]

        left_hip = [landmarks[23].x * w, landmarks[23].y * h]
        left_knee = [landmarks[25].x * w, landmarks[25].y * h]
        left_ankle = [landmarks[27].x * w, landmarks[27].y * h]

        # Calculate angles
        arm_angle = get_angle(left_shoulder, left_elbow, left_wrist)
        leg_angle = get_angle(left_hip, left_knee, left_ankle)

        # Display angles for debugging
        cv2.putText(frame, f"Arm: {int(arm_angle)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Leg: {int(leg_angle)}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Jumping jack detection logic
        if arm_angle > 150 and leg_angle > 160:
            stage = "up"
        if arm_angle < 70 and leg_angle < 160 and stage == "up":
            stage = "down"
            rep_count += 1

    # Display rep count
    cv2.putText(frame, f"Reps: {rep_count}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Jumping Jack Counter", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
