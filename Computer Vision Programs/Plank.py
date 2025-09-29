import cv2
import threading
import time
from queue import Queue, Empty
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import mediapipe as mp
import logging
from contextlib import suppress
from pygrabber.dshow_graph import FilterGraph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
CAM_WIDTH = 580
CAM_HEIGHT = 360
MODEL_CLASSIFY_EVERY = 3  # classify every N frames
CLASS_QUEUE_MAX = 2
POSE_QUEUE_MAX = 2
DISPLAY_FPS_LIMIT = 60
DRAW_POSE = True
PLANK_LABEL = "plank"
MODEL_SIZE = 344

# ------------------------------------------------------------
# Device setup
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

processor = AutoImageProcessor.from_pretrained(
    "prithivMLmods/Gym-Workout-Classifier-SigLIP2",
    trust_remote_code=True
)
model = AutoModelForImageClassification.from_pretrained(
    "prithivMLmods/Gym-Workout-Classifier-SigLIP2",
    trust_remote_code=True
)
model.to(device).eval()
torch.backends.cudnn.benchmark = True

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------------------------------------
# Queues and state
# ------------------------------------------------------------
capture_queue = Queue(maxsize=1)
class_queue = Queue(maxsize=CLASS_QUEUE_MAX)

state = {
    "current_label": None,
    "plank_start_time": None,
    "plank_duration": 0.0,
    "pose_landmarks": None
}
state_lock = threading.Lock()
stop_event = threading.Event()

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def warmup_model():
    logging.info("Warming up model...")
    dummy = Image.new("RGB", (MODEL_SIZE, MODEL_SIZE), color=(128,128,128))
    inputs = processor(images=dummy, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        if device.type == "cuda":
            from torch.cuda.amp import autocast
            with autocast():
                _ = model(**inputs)
        else:
            _ = model(**inputs)
    logging.info("Model warmup complete.")

def detect_camera(keyword="OBS"):
    """
    Detect camera index containing the keyword.
    Fallback to 0 if none found.
    """
    graph = FilterGraph()
    for idx, name in enumerate(graph.get_input_devices()):
        logging.info(f"Camera {idx}: {name}")
        if keyword.lower() in name.lower():
            logging.info(f"Selected camera '{name}' at index {idx}")
            return idx
    logging.warning(f"No camera containing '{keyword}' found. Using default index 0")
    return 0

# ------------------------------------------------------------
# Capture thread
# ------------------------------------------------------------
def capture_thread_fn(cam_index=0):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        logging.error(f"Cannot open camera index {cam_index}")
        stop_event.set()
        return
    logging.info(f"Capture thread started (index {cam_index})")
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Frame read failed; stopping")
                stop_event.set()
                break
            with suppress(Exception):
                if capture_queue.full():
                    capture_queue.get_nowait()
                capture_queue.put_nowait(frame)
    finally:
        cap.release()
        logging.info("Capture thread exiting.")

# ------------------------------------------------------------
# Pose thread
# ------------------------------------------------------------
def pose_thread_fn():
    logging.info("Pose thread started")
    try:
        while not stop_event.is_set():
            try:
                frame = capture_queue.get(timeout=0.1)
            except Empty:
                continue
            small = cv2.resize(frame, (CAM_WIDTH//2, CAM_HEIGHT//2))
            image_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            with state_lock:
                state["pose_landmarks"] = results.pose_landmarks
            with suppress(Exception):
                if not class_queue.full():
                    class_queue.put_nowait(frame)
    finally:
        logging.info("Pose thread exiting")

# ------------------------------------------------------------
# Classification thread
# ------------------------------------------------------------
def classification_thread_fn():
    logging.info("Classification thread started")
    warmup_model()
    frame_idx = 0
    try:
        while not stop_event.is_set():
            try:
                frame = class_queue.get(timeout=0.1)
            except Empty:
                continue
            frame_idx += 1
            if MODEL_CLASSIFY_EVERY > 1 and (frame_idx % MODEL_CLASSIFY_EVERY) != 0:
                continue
            h, w = frame.shape[:2]
            short_edge = min(h, w)
            start_x = (w - short_edge) // 2
            start_y = (h - short_edge) // 2
            cropped = frame[start_y:start_y+short_edge, start_x:start_x+short_edge]
            resized = cv2.resize(cropped, (MODEL_SIZE, MODEL_SIZE))
            image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(image_rgb)
            inputs = processor(images=pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                if device.type == "cuda":
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
            logits = outputs.logits
            predicted_idx = logits.argmax(-1).item()
            label = model.config.id2label[predicted_idx]
            now = time.time()
            with state_lock:
                prev_label = state["current_label"]
                state["current_label"] = label
                if label == PLANK_LABEL:
                    if state["plank_start_time"] is None:
                        state["plank_start_time"] = now
                    state["plank_duration"] = now - state["plank_start_time"]
                else:
                    if prev_label == PLANK_LABEL and state["plank_duration"] > 0:
                        logging.info(f"Last plank duration: {state['plank_duration']:.1f}s")
                    state["plank_start_time"] = None
                    state["plank_duration"] = 0.0
    finally:
        logging.info("Classification thread exiting")

# ------------------------------------------------------------
# Display thread
# ------------------------------------------------------------
def display_thread_fn():
    logging.info("Display thread started")
    try:
        while not stop_event.is_set():
            start = time.time()
            try:
                frame = capture_queue.get(timeout=0.1)
            except Empty:
                continue
            with state_lock:
                pose_landmarks = state["pose_landmarks"]
                label = state["current_label"] or "..."
                plank_duration = state["plank_duration"]
            if DRAW_POSE and pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
            text_label = PLANK_LABEL if label == PLANK_LABEL else "not plank"
            cv2.putText(frame, f"Pose: {text_label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            if plank_duration > 0:
                cv2.putText(frame, f"Plank Time: {plank_duration:.1f}s", (10,70), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
            cv2.imshow("Plank Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
            elapsed = time.time() - start
            min_frame_time = 1.0 / DISPLAY_FPS_LIMIT
            if elapsed < min_frame_time:
                time.sleep(min_frame_time - elapsed)
    finally:
        cv2.destroyAllWindows()
        logging.info("Display thread exiting")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    cam_index = detect_camera("OBS")  # auto-detect OBS/DroidCam
    threads = [
        threading.Thread(target=capture_thread_fn, args=(cam_index,), name="capture", daemon=True),
        threading.Thread(target=pose_thread_fn, name="pose", daemon=True),
        threading.Thread(target=classification_thread_fn, name="classify", daemon=True),
        threading.Thread(target=display_thread_fn, name="display", daemon=True)
    ]
    for t in threads: t.start()
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt -> stopping")
        stop_event.set()
    logging.info("Waiting for threads to stop...")
    for t in threads:
        t.join(timeout=2.0)
    logging.info("All threads stopped. Exiting.")

if __name__ == "__main__":
    main()
