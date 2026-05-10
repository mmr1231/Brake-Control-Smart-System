
import cv2
import numpy as np
import os
import sys
import pickle
import sqlite3
import logging
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
from datetime import datetime, timedelta
import time
from PIL import Image, ImageTk, ImageDraw, ImageFont
import face_recognition
import dlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import queue
import threading
from pathlib import Path
import collections
import shutil

# ----------------- logging setup -----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ----------------- constants -----------------
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = os.path.join(str(BASE_DIR), "models")
HAAR_DIR = os.path.join(str(BASE_DIR), "haarcascades")
EMPLOYEES_DIR = os.path.join(str(BASE_DIR), "employees_data")
THUMBNAIL_DIR = os.path.join(str(BASE_DIR), "thumbnails")
ENCODINGS_DIR = os.path.join(str(BASE_DIR), "encodings")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HAAR_DIR, exist_ok=True)
os.makedirs(EMPLOYEES_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)
os.makedirs(ENCODINGS_DIR, exist_ok=True)


# --- BEGIN: fallback get_settings() ---
def get_settings():
    # Return application settings. Replace with your persistent loader if available.
    try:
        import json, os
        cfg_path = os.path.join(os.path.dirname(__file__), "app_settings.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
    except Exception:
        pass
    return {
        "camera_path": 0,
        "recognition_model": "face_recognition",
        "use_cuda": False,
        "threshold_face_recognition": 0.45,
        "threshold_dlib": 0.36
    }
# --- END: fallback get_settings() ---


APP_TITLE = "نظام مراقبة الاستراحات"

# ---- Project-local paths ----
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = os.path.join(str(BASE_DIR), "models")
HAAR_DIR = os.path.join(str(BASE_DIR), "haarcascades")
EMPLOYEES_DIR = os.path.join(str(BASE_DIR), "employees_data")
THUMBNAIL_DIR = os.path.join(str(BASE_DIR), "thumbnails")
# Ensure dirs
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HAAR_DIR, exist_ok=True)
os.makedirs(EMPLOYEES_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)
DB_PATH = "employees.db"
ENCODINGS_DIR = "encodings/"
THUMBNAIL_DIR = "thumbnails/"

# Create directories if not exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ENCODINGS_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

# --- Global Dlib Models ---
global_dlib_cnn_face_detector = dlib.cnn_face_detection_model_v1(os.path.join(MODELS_DIR, "mmod_human_face_detector.dat"))
global_dlib_face_detector = dlib.get_frontal_face_detector()
global_dlib_shape_predictor = dlib.shape_predictor(os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat"))
global_dlib_face_encoder_model = dlib.face_recognition_model_v1(os.path.join(MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat"))

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Thumbnail cache
_thumbnail_cache = {}

def get_employee_thumbnail(employee_id, size=(120,120)):
    key = f"thumb_{employee_id}_{size[0]}x{size[1]}"
    if key in _thumbnail_cache:
        return _thumbnail_cache[key]

    img_path = Path(THUMBNAIL_DIR) / f"{employee_id}.jpg"
    if img_path.exists():
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail(size, Image.Resampling.LANCZOS)
            _thumbnail_cache[key] = img
            return img
        except Exception as e:
            logger.error(f"Error loading thumbnail for {employee_id}: {e}")
    
    # Placeholder image if not found
    img = Image.new("RGB", size, (50,50,60))
    draw = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.truetype("arial.ttf", size[0]//4) # Adjust font size dynamically
    except IOError:
        fnt = ImageFont.load_default() # Fallback font
    draw.text((size[0]//3, size[1]//3), str(employee_id)[:3].upper(), font=fnt, fill=(200,200,200))
    _thumbnail_cache[key] = img
    return img

# Initialize multiple face recognition models
class MultiModelFaceRecognizer:
    def __init__(self):
        # Separate storage for each model's encodings/weights
        self.face_recognition_encodings = {}
        self.lbph_model = cv2.face.LBPHFaceRecognizer_create()
        self.dlib_descriptors = {}  # Store face descriptors for Dlib
        self.dlib_labels = []       # Store corresponding labels for Dlib
        self.svm_model = SVC(probability=True)
        self.label_encoder = LabelEncoder()
        self.current_model = "face_recognition" # Default model

        # Initialize Dlib models using the helper function
        self.dlib_detector = global_dlib_face_detector
        try:
            self.dlib_predictor = global_dlib_shape_predictor
            self.dlib_encoder = global_dlib_face_encoder_model
        except Exception as e: # Catch generic Exception as dlib errors can be varied
            logger.error(f"Error loading Dlib models: {e}")
            self.dlib_predictor = None
            self.dlib_encoder = None

        # Optional face_recognition
        self.use_face_recognition_available = False
        try:
            import face_recognition
            self.face_recognition_lib = face_recognition
            self.use_face_recognition_available = True
        except ImportError:
            logger.warning("face_recognition library not found. Using fallback methods.")
            self.face_recognition_lib = None
            self.use_face_recognition_available = False

        self.settings = get_settings() # Load initias settings
        self.lettings = get_settings() # Load initial settings
        self.load_models()

    def load_models(self):
        """Load all models and encodings"""
        try:
            # Load face recognition encodings
            encodings_file = os.path.join(ENCODINGS_DIR, "face_recognition_encodings.pkl")
            if os.path.exists(encodings_file):
                with open(encodings_file, 'rb') as f:
                    self.face_recognition_encodings = pickle.load(f)
                logger.info("Face recognition encodings loaded successfully")

            # Load LBPH model
            lbph_file = os.path.join(MODELS_DIR, "lbph_model.yml")
            if os.path.exists(lbph_file):
                self.lbph_model.read(lbph_file)
                logger.info("LBPH model loaded successfully")

            # Load Dlib descriptors and labels
            dlib_descriptors_file = os.path.join(ENCODINGS_DIR, "dlib_descriptors.pkl")
            dlib_labels_file = os.path.join(ENCODINGS_DIR, "dlib_labels.pkl")
            if os.path.exists(dlib_descriptors_file) and os.path.exists(dlib_labels_file):
                with open(dlib_descriptors_file, 'rb') as f:
                    self.dlib_descriptors = pickle.load(f)
                with open(dlib_labels_file, 'rb') as f:
                    self.dlib_labels = pickle.load(f)
                logger.info("Dlib descriptors and labels loaded successfully")

            # Load SVM model and label encoder
            svm_model_file = os.path.join(MODELS_DIR, "svm_model.pkl")
            label_encoder_file = os.path.join(ENCODINGS_DIR, "label_encoder.pkl")
            if os.path.exists(svm_model_file) and os.path.exists(label_encoder_file):
                with open(svm_model_file, 'rb') as f:
                    self.svm_model = pickle.load(f)
                with open(label_encoder_file, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info("SVM model and label encoder loaded successfully")

            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def save_models(self):
        """Save all models and encodings"""
        try:
            # Save face recognition encodings
            encodings_file = os.path.join(ENCODINGS_DIR, "face_recognition_encodings.pkl")
            with open(encodings_file, 'wb') as f:
                pickle.dump(self.face_recognition_encodings, f)
            logger.info("Face recognition encodings saved successfully")

            # Save LBPH model
            lbph_file = os.path.join(MODELS_DIR, "lbph_model.yml")
            self.lbph_model.save(lbph_file)
            logger.info("LBPH model saved successfully")

            # Save Dlib descriptors and labels
            dlib_descriptors_file = os.path.join(ENCODINGS_DIR, "dlib_descriptors.pkl")
            dlib_labels_file = os.path.join(ENCODINGS_DIR, "dlib_labels.pkl")
            with open(dlib_descriptors_file, 'wb') as f:
                pickle.dump(self.dlib_descriptors, f)
            with open(dlib_labels_file, 'wb') as f:
                pickle.dump(self.dlib_labels, f)
            logger.info("Dlib descriptors and labels saved successfully")

            # Save SVM model and label encoder
            svm_model_file = os.path.join(MODELS_DIR, "svm_model.pkl")
            label_encoder_file = os.path.join(ENCODINGS_DIR, "label_encoder.pkl")
            with open(svm_model_file, 'wb') as f:
                pickle.dump(self.svm_model, f)
            with open(label_encoder_file, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            logger.info("SVM model and label encoder saved successfully")

            logger.info("All models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def train_models(self, faces, labels, model_type="all"):
        """Train all models simultaneously in the background"""
        def _train_in_background():
            try:
                logger.info("Starting background training for all models...")

                if not faces:
                    logger.warning("No faces provided for training. Aborting.")
                    return

                # --- Pre-computation for efficiency ---
                gray_faces = [cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) for face in faces]
                resized_gray_faces = [cv2.resize(face, (200, 200)) for face in gray_faces]

                # Train face_recognition model (store encodings)
                if self.use_face_recognition_available:
                    new_encodings = {}
                    for face_img, label in zip(faces, labels):
                        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        # Since we have a cropped face, define its location as the whole image
                        h, w, _ = rgb_face.shape
                        face_location = [(0, w, h, 0)]
                        face_encodings = self.face_recognition_lib.face_encodings(rgb_face, known_face_locations=face_location)
                        if face_encodings:
                            if label not in new_encodings:
                                new_encodings[label] = []
                            new_encodings[label].append(face_encodings[0])

                    # Update existing encodings
                    for label, encs in new_encodings.items():
                        if label in self.face_recognition_encodings:
                            self.face_recognition_encodings[label].extend(encs)
                        else:
                            self.face_recognition_encodings[label] = encs
                    logger.info("face_recognition model trained successfully")

                # Convert string labels to integers for LBPH
                unique_labels = list(set(labels))
                label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
                int_labels = [label_to_int[label] for label in labels]
                int_labels = np.array(int_labels, dtype=np.int32)

                # Train LBPH model
                self.lbph_model.train(np.array(resized_gray_faces), int_labels)
                logger.info("LBPH model trained successfully")

                # --- Dlib and SVM Training (using efficient descriptors) ---
                all_descriptors = []
                all_descriptor_labels = []

                if self.dlib_predictor is not None and self.dlib_encoder is not None:
                    new_descriptors = {}
                    for face_img, gray_img, label in zip(faces, gray_faces, labels):
                        h, w = gray_img.shape
                        # Assume the cropped image is the face, create a rect for it
                        rgb_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_rect = dlib.rectangle(0, 0, w, h)
                        landmarks = self.dlib_predictor(gray_img, face_rect)
                        descriptor = np.array(self.dlib_encoder.compute_face_descriptor(rgb_face_img, landmarks))

                        # Store for dlib direct comparison
                        if label not in new_descriptors:
                            new_descriptors[label] = []
                        new_descriptors[label].append(descriptor)

                        # Store for SVM training
                        all_descriptors.append(descriptor)
                        all_descriptor_labels.append(label)

                    # Update existing descriptors
                    for label, descs in new_descriptors.items():
                        if label not in self.dlib_descriptors:
                            self.dlib_descriptors[label] = []
                        self.dlib_descriptors[label].extend(descs)

                    logger.info("Dlib model descriptors updated successfully")

                # Train SVM model using the highly efficient Dlib descriptors
                if all_descriptors:
                    # Check if there is more than one unique class (employee)
                    if len(set(all_descriptor_labels)) > 1:
                        y_encoded = self.label_encoder.fit_transform(all_descriptor_labels)
                        X_train, X_test, y_train, y_test = train_test_split(all_descriptors, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)

                        self.svm_model.fit(X_train, y_train)

                        accuracy = self.svm_model.score(X_test, y_test)
                        logger.info(f"SVM model trained on Dlib descriptors with accuracy: {accuracy:.2f}")
                    else:
                        logger.warning("SVM training skipped: requires at least 2 employees (classes) for training.")
                else:
                    logger.warning("SVM training skipped: Not enough classes (employees) for meaningful training.")
                
                # Save all models after training
                self.save_models()
                logger.info("All models trained and saved successfully in background")

            except Exception as e:
                logger.error(f"Error during background training: {e}")

        # Start training in a background thread
        training_thread = threading.Thread(target=_train_in_background, daemon=True)
        training_thread.start()

    def recognize_face(self, image, model_type=None):
        """Recognize face using the selected model"""
        if model_type is None:
            model_type = self.current_model

        try:
            if model_type == "face_recognition" and self.use_face_recognition_available:
                # Use face_recognition model
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = self.face_recognition_lib.face_locations(rgb_image)
                face_encodings = self.face_recognition_lib.face_encodings(rgb_image, face_locations)

                if not face_locations:
                    return None, 0

                for face_enc, (top, right, bottom, left) in zip(face_encodings, face_locations):
                    best_match = None
                    best_distance = float('inf')

                    for name, known_encodings in self.face_recognition_encodings.items():
                        distances = self.face_recognition_lib.face_distance(known_encodings, face_enc)
                        min_distance = distances.min()

                        if min_distance < best_distance and min_distance < self.settings.get('face_recognition_threshold', 0.6):
                            best_distance = min_distance
                            best_match = name

                    if best_match:
                        return best_match, 1 - best_distance

                return "Unknown", 0

            elif model_type == "lbph":
                # Use LBPH model
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # This part assumes a full frame is passed, which is not the case in the current workflow.
                # The detection is done in the CameraWorker, and a cropped face is passed here.
                # If a full frame were passed, we'd detect faces first.
                h, w = gray.shape
                if h > 0 and w > 0:
                    x, y = 0, 0
                    roi = gray[y:y+h, x:x+w]
                    if self.lbph_model.getLabels().size > 0:
                        label, confidence = self.lbph_model.predict(roi)
                        if confidence < self.settings.get('lbph_threshold', 100):
                            # Ensure the label encoder has been fitted before trying to use it.
                            if hasattr(self.label_encoder, 'classes_') and label < len(self.label_encoder.classes_):
                                original_label = self.label_encoder.inverse_transform([label])[0]
                            else:
                                original_label = "Unknown"
                            return original_label, 1 - (confidence / 100)
                    else:
                        logger.warning("LBPH model not trained, cannot predict.")

                return "Unknown", 0

            elif model_type == "dlib" and self.dlib_predictor is not None and self.dlib_encoder is not None:
                # Use Dlib model
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.dlib_detector(gray)

                for face in faces:
                    # Extract landmarks
                    landmarks = self.dlib_predictor(gray, face)
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Compute face descriptor
                    face_descriptor = np.array(self.dlib_encoder.compute_face_descriptor(rgb_image, landmarks))
                    
                    # Compare with stored descriptors
                    best_match = None
                    best_distance = float('inf')
                    
                    for name, stored_descriptors in self.dlib_descriptors.items():
                        for stored_desc in stored_descriptors:
                            distance = np.linalg.norm(face_descriptor - stored_desc)
                            if distance < best_distance and distance < self.settings.get('dlib_threshold', 0.6):
                                best_distance = distance
                                best_match = name
                    
                    if best_match:
                        return best_match, 1 - best_distance
                    else:
                        # If no match found, return Unknown
                        return "Unknown", 0
                return "Unknown", 0

            elif model_type == "svm":
                # Use SVM model
                # Use Dlib to get a high-quality face descriptor for SVM prediction
                if self.dlib_predictor and self.dlib_encoder:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    h, w = gray.shape
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_rect = dlib.rectangle(0, 0, w, h) # Assume the input image is already a face crop
                    landmarks = self.dlib_predictor(gray, face_rect)
                    descriptor = np.array(self.dlib_encoder.compute_face_descriptor(rgb_image, landmarks))

                    # Check if the SVM model is trained
                    if hasattr(self.svm_model, 'classes_') and len(self.svm_model.classes_) > 0:
                        label_encoded = self.svm_model.predict([descriptor])[0]
                        confidence = self.svm_model.predict_proba([descriptor])[0].max()

                        if confidence >= self.settings.get('svm_threshold', 0.7):
                            # Decode label back to original name
                            original_label = self.label_encoder.inverse_transform([label_encoded])[0]
                            return original_label, confidence
                        # else, confidence is too low, return Unknown
                    else:
                        logger.warning("SVM model not trained, cannot predict.")
                        return "Unknown", 0

                return "Unknown", 0

            else:
                # Default to face_recognition if available and selected model not recognized
                if self.use_face_recognition_available:
                    return self.recognize_face(image, "face_recognition")
                else:
                    # Fallback to LBPH if face_recognition is not available
                    return self.recognize_face(image, "lbph")

        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return "Unknown", 0

# New Camera Monitor Worker Class
class CameraMonitorWorker:
    def __init__(self, camera_index, model_name, output_queue, stop_event, face_recognizer_instance):
        self.camera_index = camera_index
        self.model_name = model_name
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.face_recognizer = face_recognizer_instance
        self.scale_factor = 0.6 # Scale down frame for faster processing

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_index}")
            self.output_queue.put({'error': 'Camera not available'})
            return

        prev_frame_time = 0
        new_frame_time = 0
        fps_deque = collections.deque(maxlen=8)

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame from camera.")
                time.sleep(0.1)
                continue

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time
            fps_deque.append(fps)
            avg_fps = sum(fps_deque) / len(fps_deque)

            small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
            gray_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # Detect faces using Haar Cascade (can be replaced with dlib_detector for better accuracy)
            # Using Dlib CNN detector instead
            cnn_face_rects = global_dlib_cnn_face_detector(gray_small_frame, 0)
            faces_rects = [(face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()) for face in cnn_face_rects]

            detections = []
            for (x_s, y_s, w_s, h_s) in faces_rects:
                # Scale back up face locations for the original frame
                x = int(x_s / self.scale_factor)
                y = int(y_s / self.scale_factor)
                w = int(w_s / self.scale_factor)
                h = int(h_s / self.scale_factor)

                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue

                employee_id, confidence = self.face_recognizer.recognize_face(face_crop, self.model_name)
                
                # For display, draw rectangle and text
                color = (0, 255, 0) if employee_id != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                text = f"{employee_id} ({confidence:.2f})"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                detections.append({
                    'employee_id': employee_id,
                    'name': employee_id, # Assuming employee_id is name for simplicity here
                    'confidence': confidence,
                    'box': (x, y, w, h)
                })

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.output_queue.put({'frame_rgb': frame_rgb, 'detections': detections, 'fps': avg_fps})
            time.sleep(0.01) # Small delay to prevent 100% CPU usage

        cap.release()
        logger.info("CameraMonitorWorker stopped.")

# Initialize multi-model recognizer

class CameraWorker:
    """A dedicated, robust class for handling camera operations in a separate thread."""
    def __init__(self, settings, output_queue, stop_event, recognizer):
        self.settings = settings
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.recognizer = recognizer
        self.scale_factor = 0.5

    def get_capture_source(self):
        """يحدد مصدر الكاميرا من الإعدادات (فهرس محلي أو رابط IP/RTSP)."""
        # التأكد من أن self.settings يحتوي على أحدث الإعدادات
        self.settings = get_settings()
        
        # استخدام أول كاميرا في القائمة كافتراضي
        cameras = self.settings.get('cameras', [{'type': 'local', 'path': '0'}])
        first_camera = cameras[0] if cameras else {'type': 'local', 'path': '0'}
        camera_type = first_camera.get('type', 'local')
        camera_path = first_camera.get('path', '0')
        if camera_type == 'ip':
            logger.info(f"Attempting to open IP camera stream: {camera_path}")
            return camera_path  # Return the RTSP URL string
        else: # 'local'
            try:
                logger.info(f"Attempting to open local camera with index: {camera_path}")
                return int(camera_path) # Return the integer index
            except (ValueError, TypeError):
                logger.error(f"Invalid local camera index '{camera_path}'. Falling back to index 0.")
                return 0

    def run(self):
        """The main loop for the camera thread."""
        source = self.get_capture_source()
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.error(f"Failed to open camera source: {source}")
            self.output_queue.put({'error': 'Camera not available'})
            return

        fps_deque = collections.deque(maxlen=10)
        prev_time = time.time()

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame from camera. Reconnecting in 2s...")
                time.sleep(2)
                cap.release()
                cap = cv2.VideoCapture(source) # Attempt to reconnect
                continue

            # --- Face detection and recognition logic (from camera_worker_patch) ---
            small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
            gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            # Use Dlib's CNN detector for better accuracy
            cnn_face_rects = global_dlib_cnn_face_detector(gray_small, 0) # 0 means no upsampling
            faces = [(face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()) for face in cnn_face_rects]

            detections = []
            for (x, y, w, h) in faces: # x, y, w, h are for the small_frame
                x_orig, y_orig, w_orig, h_orig = [int(v / self.scale_factor) for v in (x, y, w, h)]
                face_crop = frame[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig]
                if face_crop.size == 0: continue
                
                name, conf = self.recognizer.recognize_face(face_crop, self.settings.get('recognition_model'))

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), color, 2)
                text = f"{name} ({conf:.2f})"
                cv2.putText(frame, text, (x_orig, y_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                detections.append({'employee_id': name, 'name': name, 'confidence': conf, 'box': (x_orig, y_orig, w_orig, h_orig)})

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            fps_deque.append(fps)
            avg_fps = sum(fps_deque) / len(fps_deque)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.output_queue.put({'frame_rgb': frame_rgb, 'detections': detections, 'fps': avg_fps})
            time.sleep(0.02)

        cap.release()
        logger.info("CameraWorker stopped.")

# --- BEGIN PATCH: Improved Monitoring (injected) ---
def get_employee_thumbnail_for_patch(employee_id, size=(120, 120)):
    """
    Helper to get thumbnails inside the patched monitor.
    This avoids potential conflicts if the original is modified.
    """
    key = f"thumb_{employee_id}_{size[0]}x{size[1]}"
    if key in _thumbnail_cache:
        return _thumbnail_cache[key]

    img_path = Path(THUMBNAIL_DIR) / f"{employee_id}.jpg"
    if img_path.exists():
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail(size, Image.Resampling.LANCZOS)
            _thumbnail_cache[key] = img
            return img
        except Exception as e:
            logger.error(f"Error loading thumbnail for {employee_id}: {e}")
    
    # Placeholder image if not found
    img = Image.new("RGB", size, (50, 50, 60))
    # Placeholder does not need text for this patch
    _thumbnail_cache[key] = img
    return img

# --- END PATCH ---
face_recognizer = MultiModelFaceRecognizer()

# ----------------- Business Logic and State Management -----------------
class AppState:
    """Manages the live state of the application."""
    def __init__(self):
        self.inside_employees = {}  # emp_id -> {name, entry_time}
        self.violators = {}         # emp_id -> {name, violation_type, start_time}

    def add_inside(self, emp_id, name):
        if emp_id not in self.inside_employees:
            self.inside_employees[emp_id] = {'name': name, 'entry_time': datetime.now()}

    def remove_inside(self, emp_id):
        return self.inside_employees.pop(emp_id, None)

    def add_violator(self, emp_id, name, v_type):
        if emp_id not in self.violators:
            self.violators[emp_id] = {'name': name, 'violation_type': v_type, 'start_time': datetime.now()}

    def remove_violator(self, emp_id):
        return self.violators.pop(emp_id, None)

    def get_all_states(self):
        return list(self.inside_employees.items()), list(self.violators.items())

app_state = AppState()

# ----------------- database setup -----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            employee_id TEXT UNIQUE NOT NULL,
            position TEXT,
            department TEXT,
            face_encoding BLOB,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT,
            name TEXT,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            location TEXT,
            status TEXT,
            duration_minutes INTEGER DEFAULT 0,
            FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department TEXT,
            allowed_start_time TEXT,
            allowed_end_time TEXT,
            location TEXT,
            max_duration INTEGER
        )
    ''')

    # New table for rest areas management
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rest_areas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            area_name TEXT NOT NULL,
            area_type TEXT,
            start_time TEXT,
            end_time TEXT,
            max_duration INTEGER,
            capacity INTEGER,
            location TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # New table for violations tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT,
            name TEXT,
            violation_start_time TIMESTAMP,
            violation_end_time TIMESTAMP,
            violation_type TEXT,
            duration_minutes INTEGER DEFAULT 0,
            location TEXT,
            resolved INTEGER DEFAULT 0,
            FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
        )
    ''')
    
    # --- Settings Table Migration and Creation ---
    # Check if the settings table exists and has the correct schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='settings'")
    table_exists = cursor.fetchone()
    
    needs_recreation = not table_exists
    if table_exists:
        try:
            # Check for a column that only exists in the new schema
            cursor.execute("SELECT id FROM settings LIMIT 1")
        except sqlite3.OperationalError:
            # The table exists but has the old schema (e.g., key/value)
            logger.warning("Old settings table schema detected. Recreating table.")
            cursor.execute("DROP TABLE settings")
            needs_recreation = True

    if needs_recreation:
        logger.info("Creating new settings table with the correct schema.")
        cursor.execute('''
            CREATE TABLE settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),                
                camera_type TEXT DEFAULT 'local',
                camera_path TEXT DEFAULT '0',
                theme TEXT DEFAULT 'dark',
                recognition_model TEXT DEFAULT 'face_recognition',
                face_recognition_threshold REAL DEFAULT 0.6,
                lbph_threshold REAL DEFAULT 100.0,
                dlib_threshold REAL DEFAULT 0.6,
                svm_threshold REAL DEFAULT 0.7
            )
        ''')
        # Insert the single row of default settings
        cursor.execute("INSERT INTO settings (id) VALUES (1)")
        conn.commit()

    # At this point, the settings table is guaranteed to have the correct schema.
    # We can now safely perform any ALTER operations if needed for future updates.
    try:
        cursor.execute("SELECT face_recognition_threshold FROM settings LIMIT 1")
        cursor.execute("SELECT camera_type FROM settings LIMIT 1") # Check for new camera columns
    except sqlite3.OperationalError:
        logger.info("Migrating settings table: adding new columns for camera.")
        try:
            cursor.execute("ALTER TABLE settings ADD COLUMN camera_type TEXT DEFAULT 'local'")
            cursor.execute("ALTER TABLE settings ADD COLUMN camera_path TEXT DEFAULT '0'")
            conn.commit()
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not add camera columns, they might exist: {e}")
        logger.info("Migrating settings table: adding new columns")

    conn.commit()
    conn.close()

# ----------------- utility functions -----------------
def apply_modern_theme():
    settings = get_settings()
    theme = settings.get('theme', 'dark')
    ctk.set_appearance_mode(theme)
    ctk.set_default_color_theme("blue")

def get_settings():
    """Get current settings from database as a dictionary."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM settings WHERE id = 1")
    settings = cursor.fetchone()
    conn.close()
    if settings:
        return dict(settings)
    else:
        # Return a default dictionary if the settings table is empty
        return {'camera_type': 'local', 'camera_path': '0', 'theme': 'dark', 'recognition_model': 'face_recognition', 'face_recognition_threshold': 0.6, 'lbph_threshold': 100.0, 'dlib_threshold': 0.6, 'svm_threshold': 0.7}

def update_settings(settings_dict):
    """Update settings in the database from a dictionary."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    set_clause = ", ".join([f"{key} = ?" for key in settings_dict])
    values = list(settings_dict.values())
    cursor.execute(f"UPDATE settings SET {set_clause} WHERE id = 1", values)
    conn.commit()
    conn.close()

def capture_face_image(employee_id):
    """Capture face images for training and save into employees_data/<employee_id>/"""
    settings = get_settings()
    camera_type = settings.get('camera_type', 'local')
    camera_path = settings.get('camera_path', '0')
    
    if camera_type == 'local':
        camera_index = int(camera_path)
    else: # ip
        camera_index = camera_path

    cap = cv2.VideoCapture(camera_index)
    face_images = []
    count = 0

    emp_dir = os.path.join(EMPLOYEES_DIR, str(employee_id))
    os.makedirs(emp_dir, exist_ok=True)

    while count < 20:  # Capture 20 images
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use Dlib CNN detector for capturing faces
        cnn_face_rects = global_dlib_cnn_face_detector(gray, 1) # Upsample once for better detection

        for face_rect in cnn_face_rects:
            rect = face_rect.rect
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            face_img = frame[y:y+h, x:x+w]
            face_images.append(face_img)
            count += 1

            # Save image into employee folder
            img_path = os.path.join(emp_dir, f"face_{count}.jpg")
            cv2.imwrite(img_path, face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Capturing. {count}/20", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Face Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save thumbnail
    try:
        if face_images:
            thumb_path = os.path.join(THUMBNAIL_DIR, f"{employee_id}.jpg")
            img = Image.fromarray(cv2.cvtColor(face_images[0], cv2.COLOR_BGR2RGB))
            img.thumbnail((200,200))
            img.save(thumb_path)
    except Exception as e:
        logger.error(f"Error saving thumbnail: {e}")

    return face_images

def train_model_after_capture(status_callback=None):
    """Automatically train ALL models after face capture in the background"""
    def _train_after_capture():
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("SELECT employee_id, name FROM employees")
            employees = cursor.fetchall()

            all_faces = []
            all_labels = []

            for emp_id, name in employees:
                # Load saved face images
                emp_dir = os.path.join(EMPLOYEES_DIR, str(emp_id))
                for i in range(1, 21):  # 20 images per employee
                    img_path = os.path.join(emp_dir, f"face_{i}.jpg")
                    if os.path.exists(img_path):
                        face_img = cv2.imread(img_path)
                        if face_img is not None:
                            all_faces.append(face_img)
                            all_labels.append(emp_id)

            if all_faces:
                face_recognizer.train_models(all_faces, all_labels, "all")
                logger.info("Auto-training for ALL models completed in background.")
                if status_callback:
                    status_callback("تم تدريب النماذج بنجاح!")
            else:
                logger.warning("No faces found for training.")
                if status_callback:
                    status_callback("لا توجد وجوه للتدريب.")
            conn.close()
        except Exception as e:
            logger.error(f"Error during auto-training: {e}")

    # Start training in a background thread
    training_thread = threading.Thread(target=_train_after_capture, daemon=True)
    training_thread.start()

def check_rest_area_status():
    """Check which rest areas are currently active"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    current_time = datetime.now().time()

    cursor.execute('''
        SELECT * FROM rest_areas
        WHERE is_active = 1
        AND start_time <= ?
        AND end_time >= ?
    ''', (current_time.strftime("%H:%M"), current_time.strftime("%H:%M")))

    active_areas = cursor.fetchall()
    conn.close()
    return active_areas

def check_employee_violation_status(employee_id):
    """Check if employee has an active violation"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM violations
        WHERE employee_id = ?
        AND resolved = 0
    ''', (employee_id,))

    violation = cursor.fetchone()
    conn.close()
    return violation

def check_employee_attendance_status(employee_id):
    """Check if employee has an active attendance record (not exited yet)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM attendance
        WHERE employee_id = ?
        AND exit_time IS NULL
    ''', (employee_id,))

    attendance = cursor.fetchone()
    conn.close()
    return attendance

def calculate_duration(start_time, end_time):
    """Calculate duration in minutes between two timestamps"""
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    duration = end_dt - start_dt
    return int(duration.total_seconds() / 60)

def start_violation(employee_id, name, location):
    """Start a violation record for an employee"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO violations (employee_id, name, violation_start_time, violation_type, location)
        VALUES (?, ?, ?, ?, ?)
    ''', (employee_id, name, datetime.now(), "دخول غير مصرح به", location))

    conn.commit()
    conn.close()

def end_violation(employee_id):
    """End an active violation for an employee"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE violations
        SET violation_end_time = ?, resolved = 1,
            duration_minutes = (strftime('%s', ?) - strftime('%s', violation_start_time)) / 60
        WHERE employee_id = ? AND resolved = 0
    ''', (datetime.now(), datetime.now(), employee_id))

    conn.commit()
    conn.close()

def start_attendance(employee_id, name, location):
    """Start an attendance record for an employee"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO attendance (employee_id, name, entry_time, location, status)
        VALUES (?, ?, ?, ?, ?)
    ''', (employee_id, name, datetime.now(), location, "Entry"))

    conn.commit()
    conn.close()

def end_attendance(employee_id, location):
    """End an active attendance record for an employee"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE attendance
        SET exit_time = ?,
            duration_minutes = (strftime('%s', ?) - strftime('%s', entry_time)) / 60,
            status = 'Exit'
        WHERE employee_id = ? AND exit_time IS NULL
    ''', (datetime.now(), datetime.now(), employee_id))

    conn.commit()
    conn.close()

def check_time_violation(employee_id, location):
    """Check if employee exceeded rest area time limit"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get employee's attendance record
    cursor.execute('''
        SELECT entry_time FROM attendance
        WHERE employee_id = ? AND exit_time IS NULL
    ''', (employee_id,))

    attendance = cursor.fetchone()

    if attendance:
        entry_time = datetime.strptime(attendance[0], "%Y-%m-%d %H:%M:%S")
        current_time = datetime.now()

        # Get rest area time limit
        cursor.execute('''
            SELECT max_duration FROM rest_areas
            WHERE location = ? AND is_active = 1
        ''', (location,))

        area_limit = cursor.fetchone()

        if area_limit:
            max_duration = area_limit[0]  # in minutes
            elapsed_time = (current_time - entry_time).total_seconds() / 60

            if elapsed_time > max_duration:
                # Start time violation
                start_violation(employee_id, employee_id, location)
                # End the attendance record
                end_attendance(employee_id, location)
                conn.close()
                return True

    conn.close()
    return False

def get_next_active_area():
    """Get the next rest area that will become active"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    current_time = datetime.now().time()

    # Get areas that start after current time, order by start time
    cursor.execute('''
        SELECT area_name, start_time, end_time, max_duration
        FROM rest_areas
        WHERE is_active = 1
        AND start_time > ?
        ORDER BY start_time
        LIMIT 1
    ''', (current_time.strftime("%H:%M"),))

    next_area = cursor.fetchone()
    conn.close()
    return next_area

def get_current_attendance_count():
    """Get the count of employees currently in rest area"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT COUNT(*) FROM attendance
        WHERE exit_time IS NULL
    ''')
    count = cursor.fetchone()[0]

    conn.close()
    return count

def get_current_violations_count():
    """Get the count of employees currently in violations"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT COUNT(*) FROM violations
        WHERE resolved = 0
    ''')
    count = cursor.fetchone()[0]

    conn.close()
    return count

# ----------------- GUI functions -----------------
class MainApp:
    def __init__(self):
        self.app = ctk.CTk()
        self.app.title(APP_TITLE)
        self.app.geometry("1400x900")

        # Apply theme
        apply_modern_theme()

        # Create main frames
        self.create_sidebar()
        self.create_main_content()

        # Load settings
        self.load_settings()

        # Update status
        self.update_status()

        # Monitoring active flag
        self.monitoring_active_flag = False

    def create_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self.app, width=200)
        self.sidebar_frame.pack(side="left", fill="y", padx=5, pady=5)

        # Add sidebar buttons
        ctk.CTkLabel(self.sidebar_frame, text="القائمة", font=("Arial", 16)).pack(pady=10)

        btn_style = {"width": 180, "height": 40, "corner_radius": 8, "font": ("Arial", 12)}

        ctk.CTkButton(self.sidebar_frame, text="📸 إضافة موظف", command=self.show_add_employee, **btn_style).pack(pady=5)
        ctk.CTkButton(self.sidebar_frame, text="🎥 المراقبة الحية", command=self.show_live_monitor, **btn_style).pack(pady=5)
        ctk.CTkButton(self.sidebar_frame, text="👥 إدارة الموظفين", command=self.show_manage_employees, **btn_style).pack(pady=5)
        ctk.CTkButton(self.sidebar_frame, text="🏢 إدارة الاستراحات", command=self.show_manage_rest_areas, **btn_style).pack(pady=5)
        ctk.CTkButton(self.sidebar_frame, text="⚙️ الإعدادات", command=self.show_settings, **btn_style).pack(pady=5)

    def create_main_content(self):
        self.main_frame = ctk.CTkFrame(self.app)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # Welcome message
        self.welcome_frame = ctk.CTkFrame(self.main_frame)
        self.welcome_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(self.welcome_frame, text="مرحباً بكم في نظام مراقبة الاستراحات", font=("Arial", 24)).pack(expand=True)

        # Status bar
        status_frame = ctk.CTkFrame(self.app, height=40)
        status_frame.pack(fill="x", side="bottom", padx=5, pady=5)
        self.status_label = ctk.CTkLabel(status_frame, text="جاهز", font=("Arial", 12))
        self.status_label.pack(pady=10)

    def load_settings(self):
        # Load settings from database
        settings = get_settings()
        theme = settings.get('theme', 'dark')
        model = settings.get('recognition_model', 'face_recognition')

        # Apply theme
        ctk.set_appearance_mode(theme)

        # Update recognizer model
        face_recognizer.current_model = model

    def update_status(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM employees")
        emp_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM rest_areas")
        area_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM violations WHERE resolved = 0")
        active_violations = cursor.fetchone()[0]

        conn.close()


    def clear_main_frame(self):
        # Stop monitoring if active
        if hasattr(self, 'monitoring_active_flag') and self.monitoring_active_flag:
            self.monitoring_active_flag = False
            # Optionally set an event to stop the thread gracefully
            if hasattr(self, 'window_closed_event'):
                self.window_closed_event.set()

        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def show_add_employee(self):
        self.clear_main_frame()

        # Create a scrollable frame
        main_frame = ctk.CTkScrollableFrame(self.main_frame)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(main_frame, text="إضافة موظف جديد", font=("Arial", 18)).pack(pady=20)

        form_frame = ctk.CTkFrame(main_frame)
        form_frame.pack(pady=20, padx=20, fill="x")

        ctk.CTkLabel(form_frame, text="الاسم:").pack(anchor="w", padx=10, pady=5)
        name_entry = ctk.CTkEntry(form_frame, width=300)
        name_entry.pack(pady=5)

        ctk.CTkLabel(form_frame, text="رقم الموظف:").pack(anchor="w", padx=10, pady=5)
        emp_id_entry = ctk.CTkEntry(form_frame, width=300)
        emp_id_entry.pack(pady=5)

        ctk.CTkLabel(form_frame, text="الوظيفة:").pack(anchor="w", padx=10, pady=5)
        position_entry = ctk.CTkEntry(form_frame, width=300)
        position_entry.pack(pady=5)

        ctk.CTkLabel(form_frame, text="القسم:").pack(anchor="w", padx=10, pady=5)
        dept_entry = ctk.CTkEntry(form_frame, width=300)
        dept_entry.pack(pady=5)

        # NO model selection for training - ALL models will be used automatically
        # We removed the model selection combo box
        ctk.CTkLabel(form_frame, text="سيتم تدريب جميع النماذج تلقائياً", font=("Arial", 12)).pack(anchor="w", padx=10, pady=5)

        def save_employee():
            name = name_entry.get()
            emp_id = emp_id_entry.get()
            position = position_entry.get()
            dept = dept_entry.get()

            if not all([name, emp_id, position, dept]):
                messagebox.showerror("خطأ", "يرجى ملء جميع الحقول")
                return

            try:
                # Capture face images
                face_images = capture_face_image(emp_id)  # images saved to employees_data/<emp_id>/ by capture function

                if len(face_images) < 10:  # Minimum 10 images
                    messagebox.showerror("خطأ", "فشل في التقاط صور كافية للوجه")
                    return

                # Save to database
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO employees (name, employee_id, position, department)
                    VALUES (?, ?, ?, ?)
                ''', (name, emp_id, position, dept))

                conn.commit()
                conn.close()

                # Auto-train ALL models in the background after saving
                # This will now train face_recognition, LBPH, Dlib, and SVM
                train_model_after_capture() 

                messagebox.showinfo("نجاح", "تم إضافة الموظف. جاري تدريب النماذج في الخلفية...")
                name_entry.delete(0, "end")
                emp_id_entry.delete(0, "end")
                position_entry.delete(0, "end")
                dept_entry.delete(0, "end")
            except sqlite3.IntegrityError:
                logger.error(f"Attempted to add duplicate employee ID: {emp_id}")
                messagebox.showerror("خطأ", f"رقم الموظف '{emp_id}' موجود بالفعل. يرجى استخدام رقم آخر.")
            except Exception as e:
                logger.exception(f"An exception occurred while adding employee '{name}' ({emp_id})")
                messagebox.showerror("خطأ", f"حدث خطأ: {str(e)}")

        ctk.CTkButton(form_frame, text="إضافة موظف", command=save_employee).pack(pady=20)

    def show_live_monitor(self):
        self.clear_main_frame()
        self.monitoring_active_flag = True

        # Create a scrollable frame
        main_frame = ctk.CTkScrollableFrame(self.main_frame)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Model selection frame (show current model from settings)
        model_frame = ctk.CTkFrame(main_frame)
        model_frame.pack(pady=5, padx=10, fill="x")

        # Load current model from settings
        current_model = get_settings().get('recognition_model', 'face_recognition')
        ctk.CTkLabel(model_frame, text=f"نموذج التعرف الحالي: {current_model}", font=("Arial", 14)).pack(side="left", padx=10, pady=10)

        # LBPH Training status
        lbph_status_frame = ctk.CTkFrame(main_frame)
        lbph_status_frame.pack(pady=5, padx=10, fill="x")

        lbph_status_label = ctk.CTkLabel(lbph_status_frame, text="حالة LBPH: ", font=("Arial", 12))
        lbph_status_label.pack(side="left", padx=10, pady=5)

        # Check if LBPH is trained
        if hasattr(face_recognizer.lbph_model, 'isTrained') and face_recognizer.lbph_model.isTrained():
            status_text = "مُدرّب"
            status_color = "green"
        else:
            status_text = "غير مُدرّب"
            status_color = "red"

        lbph_status_value = ctk.CTkLabel(lbph_status_frame, text=status_text, font=("Arial", 12), text_color=status_color)
        lbph_status_value.pack(side="left", padx=5, pady=5)

        # Dlib status
        dlib_status_frame = ctk.CTkFrame(main_frame)
        dlib_status_frame.pack(pady=5, padx=10, fill="x")

        dlib_status_label = ctk.CTkLabel(dlib_status_frame, text="حالة Dlib: ", font=("Arial", 12))
        dlib_status_label.pack(side="left", padx=10, pady=5)

        if face_recognizer.dlib_predictor is not None and face_recognizer.dlib_encoder is not None:
            status_text = "مُدرّب"
            status_color = "green"
        else:
            status_text = "غير مُدرّب"
            status_color = "red"

        dlib_status_value = ctk.CTkLabel(dlib_status_frame, text=status_text, font=("Arial", 12), text_color=status_color)
        dlib_status_value.pack(side="left", padx=5, pady=5)

        # SVM status
        svm_status_frame = ctk.CTkFrame(main_frame)
        svm_status_frame.pack(pady=5, padx=10, fill="x")

        svm_status_label = ctk.CTkLabel(svm_status_frame, text="حالة SVM: ", font=("Arial", 12))
        svm_status_label.pack(side="left", padx=10, pady=5)

        # Check if SVM is trained (has classes)
        if hasattr(face_recognizer.svm_model, 'classes_') and len(face_recognizer.svm_model.classes_) > 0:
            status_text = "مُدرّب"
            status_color = "green"
        else:
            status_text = "غير مُدرّب"
            status_color = "red"

        svm_status_value = ctk.CTkLabel(svm_status_frame, text=status_text, font=("Arial", 12), text_color=status_color)
        svm_status_value.pack(side="left", padx=5, pady=5)

        # Active rest area info
        area_info_frame = ctk.CTkFrame(main_frame)
        area_info_frame.pack(pady=5, padx=10, fill="x")

        # Store references to labels for updates
        self.active_area_label = ctk.CTkLabel(area_info_frame, text="الاستراحة النشطة: لا يوجد", font=("Arial", 18, "bold"))
        self.active_area_label.pack(pady=10)

        self.countdown_label = ctk.CTkLabel(area_info_frame, text="الوقت المتبقي: --:--:--", font=("Arial", 14))
        self.countdown_label.pack(pady=5)

        self.next_area_label = ctk.CTkLabel(area_info_frame, text="الاستراحة التالية: لا يوجد", font=("Arial", 14))
        self.next_area_label.pack(pady=5)

        # Video frame - Create a border frame to act as an "iframe"
        video_border_frame = ctk.CTkFrame(main_frame, border_width=2, border_color="blue")
        video_border_frame.pack(pady=10, padx=10, fill="both", expand=True)

        video_frame = ctk.CTkFrame(video_border_frame)
        video_frame.pack(pady=2, padx=2, fill="both", expand=True)

        video_label = ctk.CTkLabel(video_frame, text="الكاميرا الحية")
        video_label.pack(pady=10)

        # Buttons frame
        btn_frame = ctk.CTkFrame(main_frame)
        btn_frame.pack(pady=10, padx=10, fill="x")

        start_btn = ctk.CTkButton(btn_frame, text="بدأ المراقبة", width=150, height=40)
        start_btn.pack(side="left", padx=10)

        stop_btn = ctk.CTkButton(btn_frame, text="إيقاف المراقبة", width=150, height=40, state="disabled")
        stop_btn.pack(side="left", padx=10)

        # Status frame
        status_frame = ctk.CTkFrame(main_frame)
        status_frame.pack(pady=10, padx=10, fill="x")

        status_label = ctk.CTkLabel(status_frame, text="الحالة: جاهز", font=("Arial", 14))
        status_label.pack(pady=10)

        # Notes frame - Two parts
        notes_frame = ctk.CTkFrame(main_frame)
        notes_frame.pack(pady=10, padx=10, fill="x")

        # Part 1: Inside rest area currently
        inside_frame = ctk.CTkFrame(notes_frame)
        inside_frame.pack(side="left", padx=5, fill="both", expand=True)

        ctk.CTkLabel(inside_frame, text="داخل الاستراحة حالياً", font=("Arial", 14, "bold")).pack(pady=5)
        self.inside_count_label = ctk.CTkLabel(inside_frame, text="0", font=("Arial", 16, "bold"), text_color="green")
        self.inside_count_label.pack(pady=5)

        # Part 2: Inside violations currently
        violations_frame = ctk.CTkFrame(notes_frame)
        violations_frame.pack(side="right", padx=5, fill="both", expand=True)

        ctk.CTkLabel(violations_frame, text="داخل المخالفات حالياً", font=("Arial", 14, "bold")).pack(pady=5)
        self.violations_count_label = ctk.CTkLabel(violations_frame, text="0", font=("Arial", 16, "bold"), text_color="red")
        self.violations_count_label.pack(pady=5)

        # Attendance log
        log_frame = ctk.CTkFrame(main_frame)
        log_frame.pack(pady=10, padx=10, fill="both", expand=True)

        log_text = ctk.CTkTextbox(log_frame, width=400, height=200)
        log_text.pack(pady=10, padx=10, fill="both", expand=True)

        # إضافة Queue لنقل البيانات من الخيط إلى واجهة المستخدم
        log_queue = queue.Queue()

        def log_to_queue(message):
            """Put log message into queue"""
            log_queue.put(message)

        def update_log_display():
            """Get messages from queue and update the textbox"""
            if not self.monitoring_active_flag:
                return
            try:
                while True: # استمر في قراءة جميع الرسائل في الـ queue
                    message = log_queue.get_nowait()
                    log_text.insert("end", message)
                    log_text.see("end") # تمرير إلى أسفل
            except queue.Empty:
                pass # لا توجد رسائل جديدة
            # أعد استدعاء نفسك بعد 100 مللي ثانية
            if self.monitoring_active_flag:
                self.app.after(100, update_log_display) # Update only if monitoring is active

        # Monitor thread and control variables
        monitor_thread = None
        self.window_closed_event = threading.Event()
        monitoring_active = False

        def monitor_faces():
            settings = get_settings() # Get all settings
            camera_index = settings.get('camera_index', 0)
            selected_model = settings.get('recognition_model', 'face_recognition')
            cap = cv2.VideoCapture(camera_index) # Use camera index from settings

            while not self.window_closed_event.is_set() and monitoring_active:
                ret, frame = cap.read()
                if not ret:
                    break

                # Recognize faces using selected model and its specific settings
                face_recognizer.settings = settings # Pass settings to recognizer
                name, confidence = face_recognizer.recognize_face(frame, selected_model)

                if name != "Unknown":
                    # Check if employee has active violation
                    active_violation = check_employee_violation_status(name)

                    if active_violation:
                        # Employee has active violation, check if they are exiting
                        log_to_queue(f"[{datetime.now()}] - خروج الموظف {name} من المخالفة\n")
                        end_violation(name)
                        status_label.configure(text=f"الموظف: {name} | خروج من مخالفة")
                    else:
                        # Check if employee has active attendance (not exited yet)
                        active_attendance = check_employee_attendance_status(name)

                        if active_attendance:
                            # Employee is exiting from normal attendance
                            log_to_queue(f"[{datetime.now()}] - خروج الموظف {name} من الاستراحة\n")
                            end_attendance(name, "الاستراحة")
                            status_label.configure(text=f"الموظف: {name} | خروج من الاستراحة")
                        else:
                            # Check if rest areas are active
                            active_areas = check_rest_area_status()

                            if active_areas:
                                # Check access rules
                                current_time = datetime.now().time()
                                conn = sqlite3.connect(DB_PATH)
                                cursor = conn.cursor()

                                cursor.execute('''
                                    SELECT allowed_start_time, allowed_end_time
                                    FROM access_rules
                                    WHERE department = (SELECT department FROM employees WHERE employee_id = ?)
                                ''', (name,))

                                rule = cursor.fetchone()
                                conn.close()

                                # Check if access is allowed
                                access_allowed = True
                                if rule:
                                    start_time = datetime.strptime(rule[0], "%H:%M").time()
                                    end_time = datetime.strptime(rule[1], "%H:%M").time()

                                    if not (start_time <= current_time <= end_time):
                                        access_allowed = False
                                        log_to_queue(f"[{datetime.now()}] - دخول غير مصرح به: {name}\n")
                                        # Start violation
                                        start_violation(name, name, "الاستراحة")
                                        status_label.configure(text=f"الموظف: {name} | مخالفة - دخول غير مصرح")
                                    else:
                                        # Check if employee exceeded rest area time limit
                                        if check_time_violation(name, "الاستراحة"):
                                            log_to_queue(f"[{datetime.now()}] - تجاوز وقت الاستراحة: {name}\n")
                                            status_label.configure(text=f"الموظف: {name} | مخالفة - تجاوز وقت")
                                        else:
                                            # Normal entry
                                            start_attendance(name, name, "الاستراحة")
                                            log_to_queue(f"[{datetime.now()}] - دخول: {name} (الثقة: {confidence:.2f})\n")
                                            status_label.configure(text=f"الموظف: {name} | دخول | الثقة: {confidence:.2f}")
                                else:
                                    # No access rule found, start violation
                                    start_violation(name, name, "الاستراحة")
                                    log_to_queue(f"[{datetime.now()}] - دخول غير مصرح (لا توجد قاعدة): {name}\n")
                                    status_label.configure(text=f"الموظف: {name} | مخالفة - لا توجد قاعدة")
                            else:
                                # No active rest areas, start violation
                                start_violation(name, name, "الاستراحة")
                                log_to_queue(f"[{datetime.now()}] - دخول في وقت غير نشط: {name}\n")
                                status_label.configure(text=f"الموظف: {name} | مخالفة - وقت غير نشط")

                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((800, 600))
                img_tk = ImageTk.PhotoImage(img)

                video_label.configure(image=img_tk)
                video_label.image = img_tk  # Keep reference

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

        def start_monitoring():
            nonlocal monitor_thread, monitoring_active
            current_model = get_settings().get('recognition_model', 'face_recognition')
            # Check if LBPH is trained if selected model is LBPH
            if current_model == "lbph":
                if not (hasattr(face_recognizer.lbph_model, 'isTrained') and face_recognizer.lbph_model.isTrained()):
                    messagebox.showwarning("تحذير", "نموذج LBPH غير مدرّب. لا يمكن بدء المراقبة.")
                    return
            # Check if Dlib is trained if selected model is Dlib
            elif current_model == "dlib":
                if face_recognizer.dlib_predictor is None or face_recognizer.dlib_encoder is None:
                    messagebox.showwarning("تحذير", "نموذج Dlib غير مدرّب. لا يمكن بدء المراقبة.")
                    return
            # Check if SVM is trained if selected model is SVM
            elif current_model == "svm":
                if not (hasattr(face_recognizer.svm_model, 'classes_') and len(face_recognizer.svm_model.classes_) > 0):
                    messagebox.showwarning("تحذير", "نموذج SVM غير مدرّب. لا يمكن بدء المراقبة.")
                    return

            if not monitoring_active:
                monitoring_active = True
                self.window_closed_event.clear() # Reset the event
                monitor_thread = threading.Thread(target=monitor_faces, daemon=True)
                monitor_thread.start()
                start_btn.configure(state="disabled")
                stop_btn.configure(state="normal")
                status_label.configure(text="الحالة: جاري المراقبة...")

        def stop_monitoring():
            nonlocal monitoring_active
            if monitoring_active:
                monitoring_active = False
                self.window_closed_event.set() # Signal the thread to stop
                start_btn.configure(state="normal")
                stop_btn.configure(state="disabled")
                status_label.configure(text="الحالة: متوقف")

        # Configure buttons
        start_btn.configure(command=start_monitoring)
        stop_btn.configure(command=stop_monitoring)

        # Start the log update loop
        update_log_display() # ابدأ تحديث السجل من الـ queue

        # Update area info - Use try-except to handle potential errors when frame is cleared
        def update_area_info():
            if not self.monitoring_active_flag:
                return
            try:
                # Only update if the labels still exist (not destroyed by changing pages)
                if hasattr(self, 'active_area_label'):
                    active_areas = check_rest_area_status()
                    if active_areas:
                        area_name = active_areas[0][1] # area_name
                        end_time_str = active_areas[0][4] # end_time
                        self.active_area_label.configure(text=f"الاستراحة النشطة: {area_name}")

                        # Calculate countdown
                        try:
                            end_time = datetime.strptime(end_time_str, "%H:%M").time()
                            now = datetime.now()
                            end_datetime = datetime.combine(now.date(), end_time)
                            if now > end_datetime:
                                # If end time has passed today, it means it's for tomorrow
                                end_datetime += timedelta(days=1)
                            time_diff = end_datetime - now
                            self.countdown_label.configure(text=f"الوقت المتبقي: {str(time_diff).split('.')[0]}")
                        except ValueError:
                            self.countdown_label.configure(text="الوقت المتبقي: --:--:--")
                    else:
                        self.active_area_label.configure(text="الاستراحة النشطة: لا يوجد")
                        self.countdown_label.configure(text="الوقت المتبقي: --:--:--")

                    # Get next area
                    next_area = get_next_active_area()
                    if next_area:
                        area_name, start_time, end_time, duration = next_area
                        self.next_area_label.configure(text=f"الاستراحة التالية: {area_name} (تبدأ في {start_time})")
                    else:
                        self.next_area_label.configure(text="الاستراحة التالية: لا يوجد")
            except tk.TclError:
                # Widget has been destroyed, stop updating
                return
            # Schedule next update only if monitoring is active
            if self.monitoring_active_flag:
                self.app.after(1000, update_area_info) # Update every second

        # Update counts - Use try-except to handle potential errors when frame is cleared
        def update_counts():
            if not self.monitoring_active_flag:
                return
            try:
                # Only update if the labels still exist (not destroyed by changing pages)
                if hasattr(self, 'inside_count_label'):
                    inside_count = get_current_attendance_count()
                    violations_count = get_current_violations_count()
                    self.inside_count_label.configure(text=str(inside_count))
                    self.violations_count_label.configure(text=str(violations_count))
            except tk.TclError:
                # Widget has been destroyed, stop updating
                return
            # Schedule next update only if monitoring is active
            if self.monitoring_active_flag:
                self.app.after(2000, update_counts) # Update every 2 seconds

        update_area_info()
        update_counts()

    def show_live_monitor_integrated(self):
        """
        An integrated, improved live monitor that is part of the MainApp class,
        removing the need for patching.
        """
        self.clear_main_frame()
        self.monitoring_active_flag = True

        # --- UI Layout ---
        content = ctk.CTkFrame(self.main_frame)
        content.pack(fill='both', expand=True, padx=10, pady=10)
        left = ctk.CTkFrame(content)
        left.pack(side='left', fill='both', expand=True, padx=(0, 10), pady=5)
        right = ctk.CTkFrame(content, width=360)
        right.pack(side='right', fill='y', pady=5)

        video_border = ctk.CTkFrame(left, border_width=2, corner_radius=8)
        video_border.pack(fill='both', expand=True, padx=6, pady=6)
        self.video_label = ctk.CTkLabel(video_border, text='جارٍ الاتصال بالكاميرا...', width=800, height=600)
        self.video_label.pack(fill='both', expand=True, padx=6, pady=6)

        thumb_frame = ctk.CTkFrame(video_border, width=140, height=160, corner_radius=8)
        thumb_frame.place(relx=0.78, rely=0.02)
        self.thumbnail_label = ctk.CTkLabel(thumb_frame, text='صورة الموظف')
        self.thumbnail_label.pack(fill='both', expand=True, padx=4, pady=4)

        controls = ctk.CTkFrame(left, height=50)
        controls.pack(fill='x', pady=6, padx=6)
        start_btn = ctk.CTkButton(controls, text='بدأ المراقبة', width=140)
        stop_btn = ctk.CTkButton(controls, text='إيقاف المراقبة', width=140, state='disabled')
        start_btn.pack(side='left', padx=8)
        stop_btn.pack(side='left', padx=8)

        ctk.CTkLabel(right, text='🚹 داخل الاستراحة الآن', font=('Arial', 16, 'bold')).pack(pady=(6, 2))
        self.inside_list_frame = ctk.CTkScrollableFrame(right, height=220)
        self.inside_list_frame.pack(fill='x', padx=8, pady=6)

        ctk.CTkLabel(right, text='⚠️ داخل المخالفات', font=('Arial', 16, 'bold')).pack(pady=(12, 2))
        self.violations_list_frame = ctk.CTkScrollableFrame(right, height=220)
        self.violations_list_frame.pack(fill='x', padx=8, pady=6)

        stats_frame = ctk.CTkFrame(right)
        stats_frame.pack(fill='x', padx=8, pady=10)
        self.inside_count_label = ctk.CTkLabel(stats_frame, text='داخل: 0', font=('Arial', 14))
        self.inside_count_label.pack(fill='x', pady=4)
        self.violations_count_label = ctk.CTkLabel(stats_frame, text='مخالفات: 0', font=('Arial', 14))
        self.violations_count_label.pack(fill='x', pady=4)
        self.fps_label = ctk.CTkLabel(stats_frame, text='FPS: --', font=('Arial', 12))
        self.fps_label.pack(fill='x', pady=4)

        # --- Control and State Variables ---
        self._monitor_queue = queue.Queue(maxsize=6)
        self.window_closed_event = threading.Event()
        self._monitor_thread = None

        def start_monitor():
            settings = get_settings()
            face_recognizer.settings = settings # Pass settings to recognizer
            start_btn.configure(state='disabled')
            stop_btn.configure(state='normal')
            self.window_closed_event.clear()
            # Use the new CameraWorker class
            camera_worker = CameraWorker(settings, self._monitor_queue, self.window_closed_event, face_recognizer)
            self._monitor_thread = threading.Thread(target=camera_worker.run,
                daemon=True
            )
            self._monitor_thread.start()
            ui_updater() # Start the UI updater loop

        def stop_monitor():
            stop_btn.configure(state='disabled')
            start_btn.configure(state='normal')
            self.window_closed_event.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=1.0)
            self.thumbnail_label.configure(image=None, text="صورة الموظف")
            self.video_label.configure(image=None, text="تم إيقاف المراقبة")

        start_btn.configure(command=start_monitor)
        stop_btn.configure(command=stop_monitor)

        def ui_updater():
            if self.window_closed_event.is_set():
                return

            try:
                payload = self._monitor_queue.get_nowait()
                frame_rgb = payload.get('frame_rgb')
                detections = payload.get('detections', [])
                fps = payload.get('fps', None)

                # Update video feed
                if frame_rgb is not None:
                    img = Image.fromarray(frame_rgb)
                    img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 600))
                    self.video_label.configure(image=img_tk, text="")
                    self.video_label.image = img_tk

                # Process detections and update state
                recognized_ids = {det['employee_id'] for det in detections if det['employee_id']}
                
                # Handle exits
                exited_employees = set(app_state.inside_employees.keys()) - recognized_ids
                for emp_id in exited_employees:
                    end_attendance(emp_id, "الاستراحة")
                    app_state.remove_inside(emp_id)

                exited_violators = set(app_state.violators.keys()) - recognized_ids
                for emp_id in exited_violators:
                    end_violation(emp_id)
                    app_state.remove_violator(emp_id)

                # Handle entries
                last_recog = None
                for det in detections:
                    emp_id = det.get('employee_id')
                    if not emp_id: continue
                    last_recog = emp_id
                    
                    if emp_id in app_state.inside_employees or emp_id in app_state.violators:
                        continue # Already inside

                    active_areas = check_rest_area_status()
                    if active_areas:
                        start_attendance(emp_id, det['name'], "الاستراحة")
                        app_state.add_inside(emp_id, det['name'])
                    else:
                        start_violation(emp_id, det['name'], "الاستراحة")
                        app_state.add_violator(emp_id, det['name'], "دخول في وقت غير نشط")

                # Update UI lists from state
                inside_items, violator_items = app_state.get_all_states()
                
                render_list(self.inside_list_frame, inside_items)
                render_list(self.violations_list_frame, violator_items)

                self.inside_count_label.configure(text=f'داخل: {len(inside_items)}')
                self.violations_count_label.configure(text=f'مخالفات: {len(violator_items)}')
                if fps is not None:
                    self.fps_label.configure(text=f'FPS: {fps:.1f}')
                
                if last_recog:
                    thumb_img = get_employee_thumbnail_for_patch(last_recog, size=(140, 140))
                    big_thumb = ctk.CTkImage(light_image=thumb_img, dark_image=thumb_img, size=(140, 140))
                    self.thumbnail_label.configure(image=big_thumb, text='') # Update thumbnail
                    self.thumbnail_label.image = big_thumb

            except queue.Empty:
                pass # No new data, just wait for the next cycle
            except Exception as e:
                logger.error(f"UI Updater error: {e}")
            
            self.app.after(100, ui_updater)

        def render_list(frame, items):
            for w in frame.winfo_children():
                w.destroy()
            for emp_id, data in items:
                row = ctk.CTkFrame(frame, height=60)
                row.pack(fill='x', pady=4, padx=4)
                thumb_img = get_employee_thumbnail_for_patch(emp_id, size=(56, 56))
                thumb = ctk.CTkImage(light_image=thumb_img, dark_image=thumb_img, size=(56,56))
                img_lbl = ctk.CTkLabel(row, width=56, height=56, image=thumb, text="")
                img_lbl.pack(side='left', padx=6)
                ctk.CTkLabel(row, text=f"{data['name']}\nID: {emp_id}", anchor='w', justify='left').pack(side='left', padx=8)


    def show_manage_employees(self):
        self.clear_main_frame()

        # Create a scrollable frame
        main_frame = ctk.CTkScrollableFrame(self.main_frame)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Treeview for employees
        tree_frame = ctk.CTkFrame(main_frame)
        tree_frame.pack(pady=10, padx=10, fill="both", expand=True)

        columns = ("ID", "Name", "Employee ID", "Position", "Department")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=15)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)

        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def load_employees():
            for item in tree.get_children():
                tree.delete(item)

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("SELECT id, name, employee_id, position, department FROM employees")
            employees = cursor.fetchall()

            for emp in employees:
                tree.insert("", "end", values=emp)

            conn.close()

        def delete_employee():
            selected = tree.selection()
            if not selected:
                messagebox.showwarning("تحذير", "يرجى اختيار موظف للحذف")
                return

            item = tree.item(selected[0])
            emp_id = item['values'][2]  # Employee ID

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM employees WHERE employee_id = ?", (emp_id,))
            conn.commit()
            conn.close()

            load_employees()
            # Auto-train ALL models after deletion in the background
            train_model_after_capture() # This will now re-train all models
            messagebox.showinfo("نجاح", "تم حذف الموظف. جاري تحديث النماذج...")

        def update_employee():
            selected = tree.selection()
            if not selected:
                messagebox.showwarning("تحذير", "يرجى اختيار موظف للتحديث")
                return

            item = tree.item(selected[0])
            emp_id = item['values'][2]

            # Open update window in same frame
            # Clear main frame first
            self.clear_main_frame()

            # Create a scrollable frame
            update_frame = ctk.CTkScrollableFrame(self.main_frame)
            update_frame.pack(fill="both", expand=True, padx=10, pady=10)

            ctk.CTkLabel(update_frame, text="تحديث موظف", font=("Arial", 18)).pack(pady=20)

            form_frame = ctk.CTkFrame(update_frame)
            form_frame.pack(pady=20, padx=20, fill="x")

            # Get current data
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT name, position, department FROM employees WHERE employee_id = ?", (emp_id,))
            current_data = cursor.fetchone()
            conn.close()

            # Update form
            ctk.CTkLabel(form_frame, text="الاسم:").pack(anchor="w", padx=10, pady=5)
            name_entry = ctk.CTkEntry(form_frame, width=300)
            name_entry.insert(0, current_data[0])
            name_entry.pack(pady=5)

            ctk.CTkLabel(form_frame, text="الوظيفة:").pack(anchor="w", padx=10, pady=5)
            pos_entry = ctk.CTkEntry(form_frame, width=300)
            pos_entry.insert(0, current_data[1])
            pos_entry.pack(pady=5)

            ctk.CTkLabel(form_frame, text="القسم:").pack(anchor="w", padx=10, pady=5)
            dept_entry = ctk.CTkEntry(form_frame, width=300)
            dept_entry.insert(0, current_data[2])
            dept_entry.pack(pady=5)

            def save_update():
                new_name = name_entry.get()
                new_pos = pos_entry.get()
                new_dept = dept_entry.get()

                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()

                cursor.execute('''
                    UPDATE employees
                    SET name=?, position=?, department=?
                    WHERE employee_id=?
                ''', (new_name, new_pos, new_dept, emp_id))

                conn.commit()
                conn.close()

                # Return to manage employees
                self.show_manage_employees()
                # Auto-train ALL models after update in the background
                train_model_after_capture() # This will now re-train all models
                messagebox.showinfo("نجاح", "تم تحديث بيانات الموظف. جاري تحديث النماذج...")

            ctk.CTkButton(form_frame, text="تحديث", command=save_update).pack(pady=20)

        # Buttons
        btn_frame = ctk.CTkFrame(main_frame)
        btn_frame.pack(pady=10)

        ctk.CTkButton(btn_frame, text="تحديث القائمة", command=load_employees).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="حذف موظف", command=delete_employee).pack(side="left", padx=5)

        load_employees()

    def show_manage_rest_areas(self):
        self.clear_main_frame()

        # Create a scrollable frame
        main_frame = ctk.CTkScrollableFrame(self.main_frame)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Form frame - Only name, start time, and duration
        form_frame = ctk.CTkFrame(main_frame)
        form_frame.pack(pady=10, padx=10, fill="x")

        ctk.CTkLabel(form_frame, text="إضافة/تحديث استراحة", font=("Arial", 16)).pack(pady=10)

        # Form fields
        ctk.CTkLabel(form_frame, text="اسم الاستراحة:").pack(anchor="w", padx=10, pady=5)
        name_entry = ctk.CTkEntry(form_frame, width=300)
        name_entry.pack(pady=5)

        ctk.CTkLabel(form_frame, text="وقت البدء (HH:MM):").pack(anchor="w", padx=10, pady=5)
        start_time_entry = ctk.CTkEntry(form_frame, width=200)
        start_time_entry.insert(0, "08:00")
        start_time_entry.pack(pady=5)

        ctk.CTkLabel(form_frame, text="مدة الاستراحة (HH:MM):").pack(anchor="w", padx=10, pady=5)
        duration_entry = ctk.CTkEntry(form_frame, width=200)
        duration_entry.insert(0, "01:00") # Default 1 hour in HH:MM
        duration_entry.pack(pady=5)

        # Active status
        active_var = tk.BooleanVar()
        active_check = ctk.CTkCheckBox(form_frame, text="الاستراحة نشطة", variable=active_var)
        active_check.pack(pady=5)
        active_var.set(True)

        def save_rest_area():
            name = name_entry.get()
            start_time = start_time_entry.get()
            duration_str = duration_entry.get()
            is_active = 1 if active_var.get() else 0

            if not all([name, start_time, duration_str]):
                messagebox.showerror("خطأ", "يرجى ملء جميع الحقول")
                return

            try:
                # Validate time format
                datetime.strptime(start_time, "%H:%M")
                duration_parts = duration_str.split(':')
                if len(duration_parts) != 2:
                    raise ValueError("Invalid duration format")
                duration_hours = int(duration_parts[0])
                duration_minutes = int(duration_parts[1])
                duration_total_minutes = duration_hours * 60 + duration_minutes

                # Calculate end time
                start_dt = datetime.strptime(start_time, "%H:%M")
                end_dt = start_dt + timedelta(minutes=duration_total_minutes)
                end_time = end_dt.strftime("%H:%M")

                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()

                # Check if editing existing area
                selected = tree.selection()
                if selected:
                    # Update existing
                    item = tree.item(selected[0])
                    area_id = item['values'][0]
                    cursor.execute('''
                        UPDATE rest_areas
                        SET area_name=?, start_time=?, end_time=?, max_duration=?, is_active=?
                        WHERE id=?
                    ''', (name, start_time, end_time, duration_total_minutes, is_active, area_id))
                    messagebox.showinfo("نجاح", "تم تحديث بيانات الاستراحة")
                else:
                    # Insert new
                    cursor.execute('''
                        INSERT INTO rest_areas (area_name, start_time, end_time, max_duration, is_active)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (name, start_time, end_time, duration_total_minutes, is_active))
                    messagebox.showinfo("نجاح", "تم إضافة الاستراحة")

                conn.commit()
                conn.close()

                load_rest_areas()
                clear_form()

            except ValueError as ve:
                messagebox.showerror("خطأ", f"صيغة الوقت/المدة غير صحيحة. استخدم HH:MM\n{str(ve)}")
            except Exception as e:
                logger.error(f"Error saving rest area: {e}")
                messagebox.showerror("خطأ", f"حدث خطأ: {str(e)}")

        def clear_form():
            name_entry.delete(0, "end")
            start_time_entry.delete(0, "end")
            start_time_entry.insert(0, "08:00")
            duration_entry.delete(0, "end")
            duration_entry.insert(0, "01:00")
            active_var.set(True)
            tree.selection_remove(tree.selection())

        # Buttons
        btn_frame = ctk.CTkFrame(form_frame)
        btn_frame.pack(pady=10)

        ctk.CTkButton(btn_frame, text="إضافة", command=save_rest_area, width=100).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="تحديث", command=save_rest_area, width=100).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="مسح", command=clear_form, width=100).pack(side="left", padx=5)

        # Treeview for rest areas - Only show name, start, end, duration
        tree_frame = ctk.CTkFrame(main_frame)
        tree_frame.pack(pady=10, padx=10, fill="both", expand=True)

        columns = ("ID", "الاسم", "وقت البدء", "وقت الانتهاء", "المدة (دقيقة)", "نشط")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=15)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def load_rest_areas():
            for item in tree.get_children():
                tree.delete(item)

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("SELECT id, area_name, start_time, end_time, max_duration, is_active FROM rest_areas")
            areas = cursor.fetchall()

            for area in areas:
                active_status = "نعم" if area[5] == 1 else "لا"
                tree.insert("", "end", values=(*area[:5], active_status))

            conn.close()

        def delete_rest_area():
            selected = tree.selection()
            if not selected:
                messagebox.showwarning("تحذير", "يرجى اختيار استراحة للحذف")
                return

            item = tree.item(selected[0])
            area_id = item['values'][0]  # Area ID

            if messagebox.askyesno("تأكيد", "هل أنت متأكد من حذف هذه الاستراحة؟"):
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()

                cursor.execute("DELETE FROM rest_areas WHERE id = ?", (area_id,))
                conn.commit()
                conn.close()

                load_rest_areas()
                messagebox.showinfo("نجاح", "تم حذف الاستراحة")

        def load_selected_area():
            selected = tree.selection()
            if not selected:
                return

            item = tree.item(selected[0])
            values = item['values']

            name_entry.delete(0, "end")
            name_entry.insert(0, values[1]) # name

            start_time_entry.delete(0, "end")
            start_time_entry.insert(0, values[2]) # start_time

            # Convert duration in minutes back to HH:MM
            duration_minutes = values[4] # duration in minutes
            hours = duration_minutes // 60
            minutes = duration_minutes % 60
            duration_str = f"{hours:02d}:{minutes:02d}"
            duration_entry.delete(0, "end")
            duration_entry.insert(0, duration_str)

            active_var.set(True if values[5] == "نعم" else False) # active status

        # Bind selection event
        tree.bind("<<TreeviewSelect>>", lambda e: load_selected_area())

        # Buttons frame for tree
        tree_btn_frame = ctk.CTkFrame(main_frame)
        tree_btn_frame.pack(pady=10, padx=10, fill="x")

        ctk.CTkButton(tree_btn_frame, text="تحديث القائمة", command=load_rest_areas, width=120).pack(side="left", padx=5)
        ctk.CTkButton(tree_btn_frame, text="حذف الاستراحة", command=delete_rest_area, width=120).pack(side="left", padx=5)

        load_rest_areas()

    def show_settings(self):
        self.clear_main_frame()

        # --- 1. Create Main Layout ---
        main_frame = ctk.CTkScrollableFrame(self.main_frame)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- 2. Load Current Settings from Database ---
        settings = get_settings()

        # --- 3. Define Save Settings Function ---
        def save_settings():
            """Saves settings from UI elements to the database."""
            new_settings = {
                # --- Camera Settings ---
                'camera_type': camera_type_combo.get(),
                'camera_path': camera_path_entry.get(),
                # --- Theme Settings ---
                'theme': theme_combo.get(),
                # --- Model Settings ---
                'recognition_model': model_combo.get(),
                # --- Threshold Settings ---
                'face_recognition_threshold': float(fr_thresh_entry.get()),
                'lbph_threshold': float(lbph_thresh_entry.get()),
                'dlib_threshold': float(dlib_thresh_entry.get()),
                'svm_threshold': float(svm_thresh_entry.get())
            }
            update_settings(new_settings)
            # Apply theme
            ctk.set_appearance_mode(new_settings['theme'])
            # Update recognizer model
            face_recognizer.current_model = new_settings['recognition_model']
            messagebox.showinfo("نجاح", "تم حفظ الإعدادات بنجاح")

        # --- 4. UI Elements: General Settings ---
        # --- 4.1. Main Title ---
        ctk.CTkLabel(main_frame, text="إعدادات النظام", font=("Arial", 18)).pack(pady=20)

        # --- 4.2. General Settings Frame ---
        general_frame = ctk.CTkFrame(main_frame)
        general_frame.pack(pady=10, padx=20, fill="x")
        ctk.CTkLabel(general_frame, text="الإعدادات العامة", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)

        # --- 4.3. Camera Settings Frame ---
        camera_frame = ctk.CTkFrame(general_frame)
        camera_frame.pack(fill='x', padx=5, pady=5)
        ctk.CTkLabel(camera_frame, text="إعدادات الكاميرا", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)

        # Camera Type Combo
        ctk.CTkLabel(camera_frame, text="نوع الكاميرا:").pack(anchor="w", padx=10, pady=5)
        camera_type_combo = ctk.CTkComboBox(camera_frame, values=["local", "ip"], width=200)
        camera_type_combo.set(settings.get('camera_type', 'local'))
        camera_type_combo.pack(pady=5)

        # Camera Path Entry
        ctk.CTkLabel(camera_frame, text="مسار الكاميرا (فهرس أو رابط RTSP):").pack(anchor="w", padx=10, pady=5)
        camera_path_entry = ctk.CTkEntry(camera_frame, width=400)
        camera_path_entry.insert(0, str(settings.get('camera_path', '0')))
        camera_path_entry.pack(pady=5)

        # Camera Type Change Event
        def on_camera_type_change(choice):
            if choice == 'local':
                camera_path_entry.delete(0, 'end')
                camera_path_entry.insert(0, '0')
            else: # ip
                camera_path_entry.delete(0, 'end')
                camera_path_entry.insert(0, 'rtsp://user:password@ip_address:port/stream')
        camera_type_combo.configure(command=on_camera_type_change)

        # Theme Combo
        ctk.CTkLabel(general_frame, text="الثيم:").pack(anchor="w", padx=10, pady=5)
        theme_combo = ctk.CTkComboBox(general_frame, values=["light", "dark", "system"], width=200)
        theme_combo.set(settings.get('theme', 'dark'))
        theme_combo.pack(pady=5)

        # Recognition Model Combo
        ctk.CTkLabel(general_frame, text="نموذج التعرف الافتراضي:").pack(anchor="w", padx=10, pady=5)
        model_combo = ctk.CTkComboBox(general_frame, values=["face_recognition", "lbph", "dlib", "svm"], width=200)
        model_combo.set(settings.get('recognition_model', 'face_recognition'))
        model_combo.pack(pady=5)

        # --- 5. UI Elements: Model-Specific Settings ---
        # --- 5.1. Model Settings Frame ---
        models_frame = ctk.CTkFrame(main_frame)
        models_frame.pack(pady=10, padx=20, fill="x")
        ctk.CTkLabel(models_frame, text="إعدادات النماذج", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)

        # Face Recognition Threshold Entry
        ctk.CTkLabel(models_frame, text="Face Recognition - عتبة المسافة (الأقل أفضل):").pack(anchor="w", padx=10, pady=(10,0))
        fr_thresh_entry = ctk.CTkEntry(models_frame, width=200)
        fr_thresh_entry.insert(0, str(settings.get('face_recognition_threshold', 0.6)))
        fr_thresh_entry.pack(pady=5)

        # LBPH Threshold Entry
        ctk.CTkLabel(models_frame, text="LBPH - عتبة الثقة (الأقل أفضل):").pack(anchor="w", padx=10, pady=(10,0))
        lbph_thresh_entry = ctk.CTkEntry(models_frame, width=200)
        lbph_thresh_entry.insert(0, str(settings.get('lbph_threshold', 100.0)))
        lbph_thresh_entry.pack(pady=5)

        # Dlib Threshold Entry
        ctk.CTkLabel(models_frame, text="Dlib - عتبة المسافة (الأقل أفضل):").pack(anchor="w", padx=10, pady=(10,0))
        dlib_thresh_entry = ctk.CTkEntry(models_frame, width=200)
        dlib_thresh_entry.insert(0, str(settings.get('dlib_threshold', 0.6)))
        dlib_thresh_entry.pack(pady=5)

        # SVM Threshold Entry
        ctk.CTkLabel(models_frame, text="SVM - عتبة الثقة (الأعلى أفضل):").pack(anchor="w", padx=10, pady=(10,0))
        svm_thresh_entry = ctk.CTkEntry(models_frame, width=200)
        svm_thresh_entry.insert(0, str(settings.get('svm_threshold', 0.7)))
        svm_thresh_entry.pack(pady=5)

        # --- 6. UI Elements: Manual Training Section ---
        # --- 6.1. Training Frame ---
        training_frame = ctk.CTkFrame(main_frame)
        training_frame.pack(pady=10, padx=20, fill="x")
        ctk.CTkLabel(training_frame, text="إدارة التدريب", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)

        # Training Status Label
        training_status_label = ctk.CTkLabel(training_frame, text="", font=("Arial", 12), text_color="yellow")
        training_status_label.pack(pady=5)

        # Update Training Status Function
        def update_training_status_ui(message):
            self.app.after(0, lambda: training_status_label.configure(text=message))
            if "نجاح" in message or "خطأ" in message or "لا توجد" in message:
                self.app.after(5000, lambda: training_status_label.configure(text="")) # Clear after 5 seconds

        def start_manual_retraining():
            if messagebox.askyesno("تأكيد التدريب", "هل أنت متأكد من رغبتك في إعادة تدريب جميع النماذج؟ قد يستغرق هذا بعض الوقت."):
                update_training_status_ui("جاري بدء التدريب...")
                train_model_after_capture(status_callback=update_training_status_ui) # Start training

        # Retrain Button
        ctk.CTkButton(training_frame, text="إعادة تدريب جميع النماذج", command=start_manual_retraining).pack(pady=10)

        # --- 7. Save Settings Button ---
        ctk.CTkButton(main_frame, text="حفظ الإعدادات", command=save_settings).pack(pady=20)

        self.app.mainloop()

# ----------------- startup -----------------
if __name__ == "__main__":
    init_db()
    train_model_after_capture()
    # Override the old monitor with the new integrated one
    MainApp.show_live_monitor = MainApp.show_live_monitor_integrated

    app = MainApp()
    app.run()

# --- Integration: Replace monitor page with modern CTk monitor and add 'Start Optimized' button ---
import importlib, threading, queue, time
from PIL import Image, ImageTk

def _create_new_sidebar(self):
    try:
        if hasattr(self, 'sidebar_frame'):
            self.sidebar_frame.destroy()
    except Exception:
        pass
    self.sidebar_frame = ctk.CTkFrame(self.app, width=200)
    self.sidebar_frame.pack(side="left", fill="y", padx=5, pady=5)
    ctk.CTkLabel(self.sidebar_frame, text="القائمة", font=("Arial",16)).pack(pady=10)
    btn_style = {"width":180, "height":40, "corner_radius":8, "font":("Arial",12)}
    ctk.CTkButton(self.sidebar_frame, text="📸 إضافة موظف", command=getattr(self,'show_add_employee', lambda:None), **btn_style).pack(pady=5)
    ctk.CTkButton(self.sidebar_frame, text="🎥 المراقبة الحية", command=getattr(self,'show_live_monitor_integrated', lambda:None), **btn_style).pack(pady=5)
    ctk.CTkButton(self.sidebar_frame, text="👥 إدارة الموظفين", command=getattr(self,'show_manage_employees', lambda:None), **btn_style).pack(pady=5)
    ctk.CTkButton(self.sidebar_frame, text="🏢 إدارة الاستراحات", command=getattr(self,'show_manage_rest_areas', lambda:None), **btn_style).pack(pady=5)
    ctk.CTkButton(self.sidebar_frame, text="⚙️ الإعدادات", command=getattr(self,'show_settings', lambda:None), **btn_style).pack(pady=5)

# Monkeypatch
MainApp.create_sidebar = _create_new_sidebar

def _show_live_monitor_integrated(self):
    # Clear main and build new monitor UI
    try:
        self.clear_main_frame()
    except Exception:
        pass
    frame = ctk.CTkFrame(self.main_frame)
    frame.pack(fill='both', expand=True, padx=8, pady=8)
    left = ctk.CTkFrame(frame)
    left.pack(side='left', fill='both', expand=True, padx=(0,8))
    right = ctk.CTkFrame(frame, width=340)
    right.pack(side='right', fill='y')

    # Top: active rest area + countdown
    top = ctk.CTkFrame(left)
    top.pack(fill='x', pady=6)
    self.active_area_label = ctk.CTkLabel(top, text='الاستراحة النشطة: --', font=('Arial',16,'bold'))
    self.active_area_label.pack(side='left', padx=6)
    self.countdown_label = ctk.CTkLabel(top, text='الوقت المتبقي: --:--:--', font=('Arial',14))
    self.countdown_label.pack(side='left', padx=6)

    # Video area
    video = ctk.CTkFrame(left, corner_radius=8, border_width=1)
    video.pack(fill='both', expand=True, pady=6)
    self.video_label = ctk.CTkLabel(video, text='جارٍ الاتصال بالكاميرا...', width=800, height=560)
    self.video_label.pack(fill='both', expand=True, padx=6, pady=6)

    # Controls
    controls = ctk.CTkFrame(left)
    controls.pack(fill='x', pady=6)
    self.start_opt_btn = ctk.CTkButton(controls, text='بدء المراقبة المحسنة', fg_color='#1f6feb', width=180)
    self.stop_opt_btn = ctk.CTkButton(controls, text='إيقاف المراقبة', fg_color='#ff4d4d', width=180, state='disabled')
    self.start_opt_btn.pack(side='left', padx=6)
    self.stop_opt_btn.pack(side='left', padx=6)

    # Right: inside list and violations (scrollable)
    ctk.CTkLabel(right, text='داخل الاستراحة الآن', font=('Arial',14,'bold')).pack(anchor='w', pady=(6,2), padx=6)
    self.inside_scroll = ctk.CTkScrollableFrame(right, height=220)
    self.inside_scroll.pack(fill='x', padx=6, pady=6)
    ctk.CTkLabel(right, text='مخالفات الآن', font=('Arial',14,'bold')).pack(anchor='w', pady=(8,2), padx=6)
    self.violations_scroll = ctk.CTkScrollableFrame(right, height=220)
    self.violations_scroll.pack(fill='x', padx=6, pady=6)

    # Stats
    stats = ctk.CTkFrame(right)
    stats.pack(fill='x', padx=6, pady=8)
    self.inside_count_label = ctk.CTkLabel(stats, text='داخل: 0')
    self.inside_count_label.pack(anchor='w')
    self.violations_count_label = ctk.CTkLabel(stats, text='مخالفات: 0')
    self.violations_count_label.pack(anchor='w')

    # Pipeline refs
    self._opt_pipeline = None
    self._preview_thread = None
    self._preview_stop = threading.Event()

    def _update_counts_and_panels(detections):
        # simple population: clear and add rows
        for w in list(self.inside_scroll.winfo_children()): w.destroy()
        for w in list(self.violations_scroll.winfo_children()): w.destroy()
        inside = []
        viols = []
        try:
            # try from app_state if present
            inside_items = getattr(app_state,'inside_employees', {}).items()
            viol_items = getattr(app_state,'violators', {}).items()
            for emp_id, info in inside_items:
                inside.append({'id':emp_id,'name':info.get('name')})
            for emp_id, info in viol_items:
                viols.append({'id':emp_id,'name':info.get('name')})
        except Exception:
            # fallback from detections list
            for d in detections:
                if d.get('name') and d.get('name') != 'Unknown':
                    inside.append({'id': d.get('employee_id', d.get('name')), 'name': d.get('name')})
                else:
                    viols.append({'id': d.get('employee_id', d.get('name')), 'name': d.get('name')})
        # add UI rows
        for it in inside:
            row = ctk.CTkFrame(self.inside_scroll, height=56)
            row.pack(fill='x', pady=4, padx=4)
            try:
                thumb = get_employee_thumbnail_for_patch(it['id'], size=(48,48))
                imtk = ImageTk.PhotoImage(thumb)
                lbl = ctk.CTkLabel(row, image=imtk, text='')
                lbl.image = imtk
                lbl.pack(side='left', padx=6)
            except Exception:
                pass
            ctk.CTkLabel(row, text=f"{it['name']}\nID: {it['id']}").pack(side='left', padx=6)

        for it in viols:
            row = ctk.CTkFrame(self.violations_scroll, height=56)
            row.pack(fill='x', pady=4, padx=4)
            try:
                thumb = get_employee_thumbnail_for_patch(it['id'], size=(48,48))
                imtk = ImageTk.PhotoImage(thumb)
                lbl = ctk.CTkLabel(row, image=imtk, text='')
                lbl.image = imtk
                lbl.pack(side='left', padx=6)
            except Exception:
                pass
            ctk.CTkLabel(row, text=f"{it['name']}\nID: {it['id']}").pack(side='left', padx=6)

    def _preview_loop(pipeline):
        q = getattr(pipeline, 'recognize_queue', None)
        if q is None:
            return
        import cv2
        last_time = time.time()
        fps_acc = []
        while not self._preview_stop.is_set():
            try:
                frame, results = q.get(timeout=1.0)
            except Exception:
                continue
            # annotate frame
            for rect, name, conf in results:
                l,t,r,b = rect.left(), rect.top(), rect.right(), rect.bottom()
                color = (0,255,0) if name!='Unknown' else (0,0,255)
                cv2.rectangle(frame, (l,t), (r,b), color, 2)
                cv2.putText(frame, f"{name} ({conf:.2f})", (l, max(10, t-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # update image
            try:
                im = Image.fromarray(frame)
                im = im.resize((800,560))
                imtk = ImageTk.PhotoImage(im)
                self.video_label.configure(image=imtk, text='')
                self.video_label.image = imtk
                _update_counts_and_panels([{'employee_id': r[1], 'name': r[1], 'confidence': r[2]} for r in results])
            except Exception:
                pass

    def _start_optimized():
        try:
            mod = importlib.import_module('app_optimized_full')
            OptimizedPipeline = getattr(mod, 'OptimizedPipeline', None)
        except Exception:
            OptimizedPipeline = None
        if OptimizedPipeline is not None:
            try:
                cam = get_settings().get('camera_path', '0')
                try: cam_src = int(cam)
                except: cam_src = cam
                self._opt_pipeline = OptimizedPipeline(camera_source=cam_src, recognizer=None)
                self._opt_pipeline.start()
                self._preview_stop.clear()
                self._preview_thread = threading.Thread(target=_preview_loop, args=(self._opt_pipeline,), daemon=True)
                self._preview_thread.start()
                self.start_opt_btn.configure(state='disabled'); self.stop_opt_btn.configure(state='normal')
            except Exception as e:
                messagebox.showerror('خطأ', f'فشل تشغيل النسخة المحسنة: {e}')
        else:
            messagebox.showwarning('تنبيه', 'الوحدة المحسنة غير متوفرة. تأكد من ملف app_optimized_full.py')

    def _stop_optimized():
        try:
            if self._opt_pipeline is not None:
                try: self._opt_pipeline.stop()
                except: pass
                self._opt_pipeline = None
            self._preview_stop.set()
            self.start_opt_btn.configure(state='normal'); self.stop_opt_btn.configure(state='disabled')
        except Exception:
            pass

    self.start_opt_btn.configure(command=_start_optimized)
    self.stop_opt_btn.configure(command=_stop_optimized)

    # initial update
    _update_counts_and_panels([])

# Monkeypatch method
MainApp.show_live_monitor_integrated = _show_live_monitor_integrated

# --- End of integration ---

# ------------------ Embedded Optimized Pipeline ------------------
# Lightweight multi-threaded pipeline (camera -> detect -> recognize -> render)
# This code is embedded to provide the 'optimized' option when the user clicks the new button.
import threading, queue, collections, time, numpy as np

try:
    import dlib
except Exception:
    dlib = None

try:
    import face_recognition
    FACE_RECOG_AVAILABLE = True
except Exception:
    face_recognition = None
    FACE_RECOG_AVAILABLE = False

SCALE_FACTOR = 0.4
FRAME_QUEUE_MAX = 3
CACHE_MAXLEN = 5
FACE_RECOG_THRESHOLD = 0.45

class OptimizedRecognizer:
    def __init__(self, model='face_recognition'):
        self.model = model
        self.face_db = {}  # expected to be filled by main app if available
        self.dlib_db = {}
        self.cache = collections.deque(maxlen=CACHE_MAXLEN)

    def recognize(self, face_rgb):
        # face_rgb: RGB numpy array, aligned
        # caching
        if self.cache:
            names = [r[0] for r in self.cache]
            best = max(set(names), key=names.count)
            if names.count(best) >= (len(self.cache)//2 + 1):
                for r in reversed(self.cache):
                    if r[0] == best and r[1] >= 0.8:
                        return r
        name = 'Unknown'; conf = 0.0
        try:
            if self.model == 'face_recognition' and FACE_RECOG_AVAILABLE:
                encs = face_recognition.face_encodings(face_rgb)
                if encs:
                    enc = encs[0]
                    best_dist = 1.0
                    best_name = 'Unknown'
                    for n, lst in self.face_db.items():
                        dists = face_recognition.face_distance(lst, enc)
                        if len(dists)>0:
                            m = float(np.min(dists))
                            if m < best_dist:
                                best_dist = m; best_name = n
                    if best_name != 'Unknown' and best_dist <= FACE_RECOG_THRESHOLD:
                        name = best_name; conf = 1.0 - best_dist
            # other models could be added
        except Exception:
            pass
        self.cache.append((name, conf))
        return name, conf

class OptimizedPipeline:
    def __init__(self, camera_source=0, recognizer=None):
        self.camera_source = camera_source
        self.frame_q = queue.Queue(maxsize=FRAME_QUEUE_MAX)
        self.detect_q = queue.Queue(maxsize=FRAME_QUEUE_MAX)
        self.recognize_q = queue.Queue(maxsize=FRAME_QUEUE_MAX)
        self.stop_event = threading.Event()
        self.recognizer = recognizer or OptimizedRecognizer()
        # detectors
        self.cnn = None
        if dlib is not None:
            try:
                # try to find mmod in models/ automatically
                import os
                mmod = os.path.join(os.path.dirname(__file__), 'models', 'mmod_human_face_detector.dat')
                if os.path.exists(mmod):
                    self.cnn = dlib.cnn_face_detection_model_v1(mmod)
            except Exception:
                self.cnn = None
        self.hog = dlib.get_frontal_face_detector() if dlib is not None else None
        self.threads = []

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera source")
        t1 = threading.Thread(target=self._camera_thread, daemon=True)
        t2 = threading.Thread(target=self._detect_thread, daemon=True)
        t3 = threading.Thread(target=self._recognize_thread, daemon=True)
        t1.start(); t2.start(); t3.start()
        self.threads = [t1,t2,t3]

    def stop(self):
        self.stop_event.set()
        try:
            self.cap.release()
        except Exception:
            pass

    def _camera_thread(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05); continue
            small = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA)
            try:
                self.frame_q.put((frame, small), timeout=0.5)
            except queue.Full:
                pass

    def _detect_thread(self):
        while not self.stop_event.is_set():
            try:
                frame, small = self.frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rects = []
            try:
                if self.cnn is not None:
                    dets = self.cnn(small_rgb, 0)
                    for d in dets:
                        rects.append(d.rect)
                elif self.hog is not None:
                    dets = self.hog(small_rgb, 0)
                    for r in dets:
                        rects.append(r)
            except Exception:
                pass
            # scale to original
            scaled = []
            h,w = frame.shape[:2]
            for r in rects:
                left = int(r.left()/SCALE_FACTOR); top = int(r.top()/SCALE_FACTOR)
                right = int(r.right()/SCALE_FACTOR); bottom = int(r.bottom()/SCALE_FACTOR)
                left = max(0,left); top = max(0,top); right=min(w-1,right); bottom=min(h-1,bottom)
                scaled.append(dlib.rectangle(left, top, right, bottom) if dlib is not None else None)
            try:
                self.detect_q.put((frame, scaled), timeout=0.5)
            except queue.Full:
                pass

    def _recognize_thread(self):
        while not self.stop_event.is_set():
            try:
                frame, rects = self.detect_q.get(timeout=0.5)
            except queue.Empty:
                continue
            results = []
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for rect in rects:
                if rect is None: continue
                l,t,r,b = rect.left(), rect.top(), rect.right(), rect.bottom()
                face = frame_rgb[t:b, l:r]
                if face.size==0: continue
                # optional alignment skipped for lightweight version
                name, conf = self.recognizer.recognize(face)
                results.append((rect, name, conf))
            try:
                self.recognize_q.put((frame, results), timeout=0.5)
            except queue.Full:
                pass

# ------------------ End embedded optimized pipeline ------------------



# === ENHANCEMENTS ADDED ===
# Face alignment, CLAHE, CUDA check, SVM trainer integration placeholders.
# You must merge actual logic where appropriate.

# ------------------ BEGIN ENHANCED MODULES ------------------
# Added: EnhancedRecognizer, EnhancedPipeline (alignment + CLAHE + CUDA support),
# SVM trainer utilities, and safe UI integration hooks.

import threading, queue, collections, time, os, pickle
import numpy as np

try:
    import dlib
    DLIB_AVAILABLE = True
except Exception:
    dlib = None
    DLIB_AVAILABLE = False

try:
    import face_recognition
    FACE_RECOG_AVAILABLE = True
except Exception:
    face_recognition = None
    FACE_RECOG_AVAILABLE = False

try:
    import cv2
except Exception:
    cv2 = None

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR, exist_ok=True)
MMOD_PATH = os.path.join(MODELS_DIR, 'mmod_human_face_detector.dat')
DLIB_SHAPE = os.path.join(MODELS_DIR, 'shape_predictor_68_face_landmarks.dat')
DLIB_ENCODER = os.path.join(MODELS_DIR, 'dlib_face_recognition_resnet_model_v1.dat')
SVM_MODEL = os.path.join(MODELS_DIR, 'svm_model.pkl')
SVM_LABELS = os.path.join(MODELS_DIR, 'svm_labels.pkl')

if cv2 is not None:
    try:
        CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    except Exception:
        CLAHE = None
else:
    CLAHE = None

# Try enable dlib CUDA if available
HAS_CUDA = False
try:
    if DLIB_AVAILABLE and hasattr(dlib, 'cuda') and getattr(dlib, 'DLIB_USE_CUDA', False):
        HAS_CUDA = True
except Exception:
    HAS_CUDA = False

class EnhancedRecognizer:
    def __init__(self, model='face_recognition'):
        self.model = model
        self.face_db = {}
        self.dlib_db = {}
        self.cache = collections.deque(maxlen=5)
        self.svm = None
        self.svm_labels = None
        try:
            if os.path.exists(SVM_MODEL) and os.path.exists(SVM_LABELS):
                with open(SVM_MODEL, 'rb') as f:
                    self.svm = pickle.load(f)
                with open(SVM_LABELS, 'rb') as f:
                    self.svm_labels = pickle.load(f)
        except Exception:
            self.svm = None

    def set_databases(self, face_db=None, dlib_db=None):
        if face_db is not None:
            self.face_db = face_db
        if dlib_db is not None:
            self.dlib_db = dlib_db

    def _cached(self):
        if not self.cache:
            return None
        names = [n for n,c in self.cache]
        best = max(set(names), key=names.count)
        if names.count(best) >= (len(self.cache)//2 + 1):
            for n,c in reversed(self.cache):
                if n == best and c >= 0.8:
                    return (n,c)
        return None

    def recognize(self, face_rgb):
        """
        face_rgb: aligned RGB face image (numpy array)
        returns: name, confidence
        """
        try:
            hit = self._cached()
            if hit:
                return hit
            name = 'Unknown'; conf = 0.0
            img = face_rgb
            if CLAHE is not None and cv2 is not None:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                norm = CLAHE.apply(gray)
                img = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
            if self.model == 'face_recognition' and FACE_RECOG_AVAILABLE:
                encs = face_recognition.face_encodings(img)
                if encs:
                    enc = encs[0]
                    best_name='Unknown'; best_dist=1e6
                    for n, lst in self.face_db.items():
                        if not lst: continue
                        dists = face_recognition.face_distance(lst, enc)
                        m = float(np.min(dists))
                        if m < best_dist:
                            best_dist = m; best_name = n
                    if best_name!='Unknown' and best_dist <= 0.45:
                        name = best_name; conf = 1.0 - best_dist
                        self.cache.append((name, conf))
                        return name, conf
            if self.model == 'svm' and self.svm is not None and FACE_RECOG_AVAILABLE:
                encs = face_recognition.face_encodings(img)
                if encs:
                    enc = encs[0].reshape(1, -1)
                    try:
                        probs = self.svm.predict_proba(enc)[0]
                        idx = int(np.argmax(probs))
                        if probs[idx] >= 0.85:
                            label = self.svm_labels[idx] if self.svm_labels else str(idx)
                            self.cache.append((label, float(probs[idx])))
                            return label, float(probs[idx])
                    except Exception:
                        pred = self.svm.predict(enc)[0]
                        label = self.svm_labels[pred] if self.svm_labels else str(pred)
                        self.cache.append((label, 0.5))
                        return label, 0.5
        except Exception:
            pass
        self.cache.append(('Unknown', 0.0))
        return 'Unknown', 0.0

class EnhancedPipeline:
    def __init__(self, camera_source=0, recognizer=None, scale=0.4):
        self.camera_source = camera_source
        self.scale = scale
        self.frame_q = queue.Queue(maxsize=3)
        self.detect_q = queue.Queue(maxsize=3)
        self.recognize_q = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()
        self.recognizer = recognizer or EnhancedRecognizer()
        self.cnn = None
        if DLIB_AVAILABLE and os.path.exists(MMOD_PATH):
            try:
                self.cnn = dlib.cnn_face_detection_model_v1(MMOD_PATH)
            except Exception:
                self.cnn = None
        self.hog = dlib.get_frontal_face_detector() if DLIB_AVAILABLE else None
        self.threads = []

    def start(self):
        import cv2 as _cv2
        self.cap = _cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            raise RuntimeError('Cannot open camera source')
        t1 = threading.Thread(target=self._camera_thread, daemon=True)
        t2 = threading.Thread(target=self._detect_thread, daemon=True)
        t3 = threading.Thread(target=self._recognize_thread, daemon=True)
        t1.start(); t2.start(); t3.start()
        self.threads = [t1,t2,t3]

    def stop(self):
        self.stop_event.set()
        try:
            self.cap.release()
        except Exception:
            pass

    def _camera_thread(self):
        import cv2 as _cv2
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05); continue
            small = _cv2.resize(frame, (0,0), fx=self.scale, fy=self.scale, interpolation=_cv2.INTER_AREA)
            try:
                self.frame_q.put((frame, small), timeout=0.5)
            except queue.Full:
                pass

    def _detect_thread(self):
        import cv2 as _cv2
        while not self.stop_event.is_set():
            try:
                frame, small = self.frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            small_rgb = _cv2.cvtColor(small, _cv2.COLOR_BGR2RGB)
            rects = []
            try:
                if self.cnn is not None:
                    dets = self.cnn(small_rgb, 0)
                    for d in dets:
                        rects.append(d.rect)
                elif self.hog is not None:
                    dets = self.hog(small_rgb, 0)
                    for r in dets:
                        rects.append(r)
            except Exception:
                pass
            scaled = []
            h,w = frame.shape[:2]
            for r in rects:
                left = int(r.left()/self.scale); top = int(r.top()/self.scale)
                right = int(r.right()/self.scale); bottom = int(r.bottom()/self.scale)
                left = max(0,left); top = max(0,top); right=min(w-1,right); bottom=min(h-1,bottom)
                if DLIB_AVAILABLE:
                    scaled.append(dlib.rectangle(left, top, right, bottom))
                else:
                    class SimpleRect:
                        def __init__(self,l,t,r,b):
                            self._l,self._t,self._r,self._b = l,t,r,b
                        def left(self): return self._l
                        def top(self): return self._t
                        def right(self): return self._r
                        def bottom(self): return self._b
                    scaled.append(SimpleRect(left, top, right, bottom))
            try:
                self.detect_q.put((frame, scaled), timeout=0.5)
            except queue.Full:
                pass

    def _get_aligned_face(self, frame_rgb, rect):
        try:
            if DLIB_AVAILABLE and os.path.exists(DLIB_SHAPE):
                shape = dlib.shape_predictor(DLIB_SHAPE)
                try:
                    shape_det = shape(frame_rgb, rect)
                    chip = dlib.get_face_chip(frame_rgb, shape_det, size=150)
                    return chip
                except Exception:
                    pass
        except Exception:
            pass
        l,t,r,b = rect.left(), rect.top(), rect.right(), rect.bottom()
        crop = frame_rgb[t:b, l:r]
        try:
            import cv2 as _cv2
            return _cv2.resize(crop, (150,150))
        except Exception:
            return crop

    def _recognize_thread(self):
        import cv2 as _cv2
        while not self.stop_event.is_set():
            try:
                frame, rects = self.detect_q.get(timeout=0.5)
            except queue.Empty:
                continue
            results = []
            frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            for rect in rects:
                try:
                    face_chip = self._get_aligned_face(frame_rgb, rect)
                    if face_chip is None: continue
                    name, conf = self.recognizer.recognize(face_chip)
                    results.append((rect, name, conf))
                    try:
                        if name != 'Unknown' and callable(globals().get('mark_attendance')):
                            try: mark_attendance(name)
                            except Exception: pass
                        elif name == 'Unknown' and callable(globals().get('record_violation')):
                            try: record_violation({'time': time.time(), 'type': 'unknown_face'})
                            except Exception: pass
                    except Exception:
                        pass
                except Exception:
                    continue
            try:
                self.recognize_q.put((frame, results), timeout=0.5)
            except queue.Full:
                pass

def gather_encodings_from_employees(root_dir):
    """Scan employee folders and return encodings list and labels."""
    encs = []
    labels = []
    if not FACE_RECOG_AVAILABLE:
        return [], []
    for d in os.listdir(root_dir):
        p = os.path.join(root_dir, d)
        if not os.path.isdir(p): continue
        for f in os.listdir(p):
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                try:
                    img = face_recognition.load_image_file(os.path.join(p,f))
                    locs = face_recognition.face_locations(img, model='hog')
                    if not locs: continue
                    e = face_recognition.face_encodings(img, known_face_locations=locs)[0]
                    encs.append(e); labels.append(d)
                except Exception:
                    continue
    return encs, labels

def train_svm_from_employees(root_dir, model_out=SVM_MODEL, labels_out=SVM_LABELS):
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder
    encs, labels = gather_encodings_from_employees(root_dir)
    if not encs:
        raise RuntimeError('No encodings found')
    X = np.vstack(encs)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, y)
    with open(model_out, 'wb') as f:
        pickle.dump(clf, f)
    with open(labels_out, 'wb') as f:
        pickle.dump(list(le.classes_), f)
    return clf, list(le.classes_)

# ------------------ END ENHANCED MODULES ------------------
