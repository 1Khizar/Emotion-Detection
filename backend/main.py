"""
FaceVibe API  –  FastAPI Backend
Emotion detection using the trained EmotionCNN model.

Architecture (must match training exactly):
  Conv2d(1→32, k=3) → ReLU → MaxPool2d(2)
  Conv2d(32→64, k=3) → ReLU → MaxPool2d(2)
  Flatten → Linear(6400→128) → ReLU → Linear(128→7)

Label order = ImageFolder alphabetical order:
  0:angry  1:disgust  2:fear  3:happy  4:neutral  5:sad  6:surprise
"""

import io
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  EXACT model architecture from training notebook
# ─────────────────────────────────────────────
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 10 * 10, 128)
        self.fc2   = nn.Linear(128, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ─────────────────────────────────────────────
#  Paths & constants
# ─────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
# .pth lives one level up (project root), NOT inside /backend
MODEL_PATH     = os.path.join(BASE_DIR, "..", "emotion_model.pth")
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

# Label order matches ImageFolder (alphabetical) from training
# Classes printed: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# Label order MUST match ImageFolder alphabetical order from notebook output:
# Classes: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_EMOJI  = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "neutral":  "😐",
    "sad":      "😢",
    "surprise": "😲",
}

# ─────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────
device = torch.device("cpu")
model  = EmotionCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
logger.info("✅ EmotionCNN loaded from %s", MODEL_PATH)

# ─────────────────────────────────────────────
#  Load dlib face detector (+ optional aligner)
# ─────────────────────────────────────────────
try:
    import dlib
    face_detector = dlib.get_frontal_face_detector()
    predictor: Optional[object] = None
    if os.path.exists(PREDICTOR_PATH):
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        logger.info("✅ dlib shape predictor loaded")
    else:
        logger.warning("⚠️  shape_predictor_68 not found — alignment disabled")
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logger.warning("⚠️  dlib not installed — falling back to OpenCV Haar cascade")
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    predictor = None


# ─────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="FaceVibe API",
    description="Facial emotion detection powered by a custom CNN",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  Face detection helpers
# ─────────────────────────────────────────────
def detect_faces_dlib(gray_img):
    """Returns list of dlib rectangles."""
    return face_detector(gray_img)


def detect_faces_opencv(gray_img):
    """Returns list of (x,y,w,h) tuples via Haar cascade."""
    rects = face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    return rects if len(rects) else []


def align_face_dlib(img_bgr, rect):
    """Eye-based alignment using 68-point landmarks (optional)."""
    if predictor is None:
        return img_bgr
    import dlib
    landmarks  = predictor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), rect)
    left_eye   = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye  = (landmarks.part(45).x, landmarks.part(45).y)
    dy    = right_eye[1] - left_eye[1]
    dx    = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = (
        (left_eye[0] + right_eye[0]) // 2,
        (left_eye[1] + right_eye[1]) // 2,
    )
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)
    return cv2.warpAffine(img_bgr, M, (img_bgr.shape[1], img_bgr.shape[0]))


# ─────────────────────────────────────────────
#  Core prediction logic
# ─────────────────────────────────────────────
def predict_from_image(img_bgr: np.ndarray) -> list:
    """
    Detect faces, crop them (SQUARE), and preprocess (Gray -> Resize -> /255).
    No alignment or equalization - matches training exactly.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    results = []

    # Get face bounding boxes
    if DLIB_AVAILABLE:
        rects = face_detector(gray)
        raw_boxes = [(r.left(), r.top(), r.width(), r.height()) for r in rects]
    else:
        raw_boxes = face_detector.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in raw_boxes:
        if w <= 0 or h <= 0: continue
        
        # 1. Make the crop square (best for CNNs)
        size = max(w, h)
        cx, cy = x + w//2, y + h//2
        sq_x = max(0, cx - size//2)
        sq_y = max(0, cy - size//2)
        
        # 2. Crop face from grayscale image
        face_crop = gray[sq_y:sq_y+size, sq_x:sq_x+size]
        if face_crop.size == 0: continue

        # 3. Preprocess to 48x48 normalized
        face_rs   = cv2.resize(face_crop, (48, 48))
        face_norm = face_rs / 255.0

        tensor = torch.tensor(face_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs  = torch.softmax(output, dim=1).squeeze().tolist()
            pred   = int(np.argmax(probs))

        label = EMOTION_LABELS[pred]
        results.append({
            "emotion":       label,
            "emoji":         EMOTION_EMOJI[label],
            "confidence":    round(probs[pred] * 100, 2),
            "probabilities": {EMOTION_LABELS[i]: round(p * 100, 2) for i, p in enumerate(probs)},
            "bbox":          {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        })

    return results


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return img


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model":  "EmotionCNN",
        "labels": EMOTION_LABELS,
        "dlib":   DLIB_AVAILABLE,
    }


@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    """Accept a multipart image file, return emotion predictions."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image files are supported.")
    data  = await file.read()
    img   = decode_image(data)
    faces = predict_from_image(img)
    if not faces:
        return JSONResponse({"faces": [], "count": 0, "message": "No faces detected."})
    return JSONResponse({"faces": faces, "count": len(faces)})


class Base64Request(BaseModel):
    image: str   # base64-encoded, may include data URI prefix


@app.post("/predict/base64")
def predict_base64(payload: Base64Request):
    """Accept a base64-encoded image (e.g. from webcam snapshot), return predictions."""
    try:
        b64 = payload.image
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 data.")
    img   = decode_image(data)
    faces = predict_from_image(img)
    if not faces:
        return JSONResponse({"faces": [], "count": 0, "message": "No faces detected."})
    return JSONResponse({"faces": faces, "count": len(faces)})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
