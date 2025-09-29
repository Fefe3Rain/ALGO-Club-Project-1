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


class PlankDetector:
    def __init__(self,
                 cam_width=580,
                 cam_height=360,
                 model_size=344,
                 classify_every=3,
                 display_fps=60,
                 plank_label="plank",
                 draw_pose=True):
        # ------------------------------
        # Settings
        # ------------------------------
        self.CAM_WIDTH = cam_width
        self.CAM_HEIGHT = cam_height
        self.MODEL_SIZE = model_size
        self.MODEL_CLASSIFY_EVERY = classify_every
        self.DISPLAY_FPS_LIMIT = display_fps
        self.PLANK_LABEL = plank_label
        self.DRAW_POSE = draw_pose

        # ------------------------------
        # Logging
        # ------------------------------
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
        self.logger = logging.getLogger("PlankDetector")

        # ------------------------------
        # Device Setup
        # ------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(
            "prithivMLmods/Gym-Workout-Classifier-SigLIP2", trust_remote_code=True
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            "prithivMLmods/Gym-Workout-Classifier-SigLIP2", trust_remote_code=True
        )
        self.model.to(self.device).eval()
        torch.backends.cudnn.benchmark = True

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ------------------------------
        # State
        # ------------------------------
        self.capture_queue = Queue(maxsize=1)
        self.class_queue = Queue(maxsize=2)
        self.state = {
            "current_label": None,
            "plank_start_time": None,
            "plank_duration": 0.0,
            "pose_landmarks": None
        }
        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()

    # ------------------------------
    # Public Methods
    # ------------------------------
    def start(self, cam_keyword="OBS"):
        cam_index = self.detect_camera(cam_keyword)
        self.threads = [
            threading.Thread(target=self.capture_thread_fn, args=(cam_index,), daemon=True),
            threading.Thread(target=self.pose_thread_fn, daemon=True),
            threading.Thread(target=self.classification_thread_fn, daemon=True),
            threading.Thread(target=self.display_thread_fn, daemon=True)
        ]
        for t in self.threads:
            t.start()
        self.logger.info("PlankDetector started.")
        try:
            while not self.stop_event.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt -> stopping")
            self.stop()
        self.logger.info("Waiting for threads to stop...")
        for t in self.threads:
            t.join(timeout=2.0)
        self.logger.info("All threads stopped. Exiting.")

    def stop(self):
        self.stop_event.set()

    # ------------------------------
    # Helper Functions
    # ------------------------------
    def warmup_model(self):
        self.logger.info("Warming up model...")
        dummy = Image.new("RGB", (self.MODEL_SIZE, self.MODEL_SIZE), color=(128, 128, 128))
        inputs = self.processor(images=dummy, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = self.model(**inputs)
        self.logger.info("Model warmup complete.")

    def detect_camera(self, keyword="OBS"):
        graph = FilterGraph()
        for idx, name in enumerate(graph.get_input_devices()):
            self.logger.info(f"Camera {idx}: {name}")
            if keyword.lower() in name.lower():
                self.logger.info(f"Selected camera '{name}' at index {idx}")
                return idx
        self.logger.warning(f"No camera containing '{keyword}' found. Using default index 0")
        return 0

    # ------------------------------
    # Threads
    # ------------------------------
    def capture_thread_fn(self, cam_index):
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_HEIGHT)
        if not cap.isOpened():
            self.logger.error(f"Cannot open camera index {cam_index}")
            self.stop_event.set()
            return
        self.logger.info(f"Capture thread started (index {cam_index})")
        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Frame read failed; stopping")
                    self.stop_event.set()
                    break
                with suppress(Exception):
                    if self.capture_queue.full():
                        self.capture_queue.get_nowait()
                    self.capture_queue.put_nowait(frame)
        finally:
            cap.release()
            self.logger.info("Capture thread exiting.")

    def pose_thread_fn(self):
        self.logger.info("Pose thread started")
        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.capture_queue.get(timeout=0.1)
                except Empty:
                    continue
                small = cv2.resize(frame, (self.CAM_WIDTH//2, self.CAM_HEIGHT//2))
                image_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)
                with self.state_lock:
                    self.state["pose_landmarks"] = results.pose_landmarks
                with suppress(Exception):
                    if not self.class_queue.full():
                        self.class_queue.put_nowait(frame)
        finally:
            self.logger.info("Pose thread exiting")

    def classification_thread_fn(self):
        self.logger.info("Classification thread started")
        self.warmup_model()
        frame_idx = 0
        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.class_queue.get(timeout=0.1)
                except Empty:
                    continue
                frame_idx += 1
                if self.MODEL_CLASSIFY_EVERY > 1 and (frame_idx % self.MODEL_CLASSIFY_EVERY) != 0:
                    continue
                h, w = frame.shape[:2]
                short_edge = min(h, w)
                start_x = (w - short_edge) // 2
                start_y = (h - short_edge) // 2
                cropped = frame[start_y:start_y+short_edge, start_x:start_x+short_edge]
                resized = cv2.resize(cropped, (self.MODEL_SIZE, self.MODEL_SIZE))
                image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(image_rgb)
                inputs = self.processor(images=pil, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_idx = logits.argmax(-1).item()
                label = self.model.config.id2label[predicted_idx]
                now = time.time()
                with self.state_lock:
                    prev_label = self.state["current_label"]
                    self.state["current_label"] = label
                    if label == self.PLANK_LABEL:
                        if self.state["plank_start_time"] is None:
                            self.state["plank_start_time"] = now
                        self.state["plank_duration"] = now - self.state["plank_start_time"]
                    else:
                        if prev_label == self.PLANK_LABEL and self.state["plank_duration"] > 0:
                            self.logger.info(f"Last plank duration: {self.state['plank_duration']:.1f}s")
                        self.state["plank_start_time"] = None
                        self.state["plank_duration"] = 0.0
        finally:
            self.logger.info("Classification thread exiting")

    def display_thread_fn(self):
        self.logger.info("Display thread started")
        try:
            while not self.stop_event.is_set():
                start = time.time()
                try:
                    frame = self.capture_queue.get(timeout=0.1)
                except Empty:
                    continue
                with self.state_lock:
                    pose_landmarks = self.state["pose_landmarks"]
                    label = self.state["current_label"] or "..."
                    plank_duration = self.state["plank_duration"]
                if self.DRAW_POSE and pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                    )
                text_label = self.PLANK_LABEL if label == self.PLANK_LABEL else "not plank"
                cv2.putText(frame, f"Pose: {text_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if plank_duration > 0:
                    cv2.putText(frame, f"Plank Time: {plank_duration:.1f}s", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Plank Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break
                elapsed = time.time() - start
                min_frame_time = 1.0 / self.DISPLAY_FPS_LIMIT
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)
        finally:
            cv2.destroyAllWindows()
            self.logger.info("Display thread exiting")
