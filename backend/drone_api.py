"""FastAPI backend exposing plant disease detection from webcam or Tello drone.

Start the server with:

    uvicorn backend.drone_api:app --host 0.0.0.0 --port 8000

The server creates a single global `detector` object that runs inference in the
background in a dedicated thread. Front-end clients can start or stop the
capture and periodically fetch the most recent JPEG frame and prediction.

This is deliberately lightweight – the heavy GUI / keyboard-control logic that
exists in the standalone `drone.py` file has been removed to keep the API
simple. If you need the full desktop view you can still run `python drone.py`.
"""
from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image
import sys

try:
    from djitellopy import Tello  # Optional – only required for drone mode
except ImportError:  # pragma: no cover
    Tello = None  # type: ignore  # noqa: N816

# ---------------------------------------------------------------------------
# Model setup ----------------------------------------------------------------
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent.parent / "model/plant-disease-model-complete (1).pth"
if not MODEL_PATH.exists():
    # fallback path in case space gets replaced or moved
    MODEL_PATH = Path(__file__).resolve().parent.parent / "model/plant-disease-model-complete.pth"

# The class list must match the one used during model training
CLASSES = [
    "Tomato__Late_blight",
    "Tomato_healthy",
    "Grape_healthy",
    "Orange_Haunglongbing(Citrus_greening)",
    "Soybean__healthy",
    "Squash_Powdery_mildew",
    "Potato_healthy",
    "Corn(maize)___Northern_Leaf_Blight",
    "Tomato__Early_blight",
    "Tomato_Septoria_leaf_spot",
    "Corn(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Strawberry__Leaf_scorch",
    "Peach_healthy",
    "Apple_Apple_scab",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato__Bacterial_spot",
    "Apple_Black_rot",
    "Blueberry_healthy",
    "Cherry(including_sour)___Powdery_mildew",
    "Peach__Bacterial_spot",
    "Apple_Cedar_apple_rust",
    "Tomato_Target_Spot",
    "Pepper,_bell__healthy",
    "Grape__Leaf_blight(Isariopsis_Leaf_Spot)",
    "Potato__Late_blight",
    "Tomato__Tomato_mosaic_virus",
    "Strawberry__healthy",
    "Apple_healthy",
    "Grape_Black_rot",
    "Potato__Early_blight",
    "Cherry_(including_sour)__healthy",
    "Corn(maize)__Common_rust",
    "Grape__Esca(Black_Measles)",
    "Raspberry__healthy",
    "Tomato_Leaf_Mold",
    "Tomato__Spider_mites Two-spotted_spider_mite",
    "Pepper,bell_Bacterial_spot",
    "Corn(maize)___healthy",
]

# --------------------------- Model architecture ---------------------------

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_acc = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_acc}


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(4))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Register ResNet9 under '__main__' so that torch.load pickles saved with
# that reference can resolve it (common when training via a notebook or
# script where __name__ == "__main__").
sys.modules.setdefault("__main__", sys.modules[__name__])
setattr(sys.modules["__main__"], "ResNet9", ResNet9)

# --------------------------- End architecture ----------------------------

# ---------------------------------------------------------------------------
# Helper functions -----------------------------------------------------------
# ---------------------------------------------------------------------------

TRANSFORM = transforms.Compose([transforms.ToTensor()])


def preprocess_image(frame: np.ndarray) -> torch.Tensor:
    """Convert OpenCV BGR frame to a model-ready tensor."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (256, 256))
    pil_img = Image.fromarray(frame_resized)
    return TRANSFORM(pil_img).unsqueeze(0)  # (1, 3, 256, 256)


# ---------------------------------------------------------------------------
# Detector class -------------------------------------------------------------
# ---------------------------------------------------------------------------


class Detector:
    """Continuously grab frames from source and run predictions in background."""

    def __init__(self, source: str = "webcam") -> None:
        self.source = source  # 'webcam' or 'tello'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The standalone drone.py file already contains the model definition –
        # copy just enough to load the model *via* torch.load without relying on
        # the original globals.
        try:
            loaded = torch.load(MODEL_PATH, map_location=self.device)
            if isinstance(loaded, nn.Module):
                self.model = loaded
            else:
                # assume state_dict-like
                state_dict = loaded["state_dict"] if "state_dict" in loaded else loaded
                self.model = ResNet9(3, len(CLASSES))
                self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Failed to load model: {e}")
            # final fallback: instantiate fresh model (will output random)
            self.model = ResNet9(3, len(CLASSES))
        self.model.eval()

        # Runtime state
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_prediction: Optional[str] = None

        # Video capture objects
        self.cap: Optional[cv2.VideoCapture] = None
        self.tello = None

    # ---------------------------------------------------------------------
    # Video helpers -------------------------------------------------------
    # ---------------------------------------------------------------------

    def _open_source(self) -> bool:
        if self.source == "webcam":
            self.cap = cv2.VideoCapture(0)
            return self.cap.isOpened()
        elif self.source == "tello":
            if Tello is None:
                print("djitellopy not installed; falling back to webcam")
                return False
            try:
                self.tello = Tello()
                self.tello.connect()
                self.tello.streamon()
                return True
            except Exception as exc:  # pragma: no cover
                print(f"Failed to init Tello: {exc}")
                return False
        return False

    def _read_frame(self) -> Optional[np.ndarray]:
        if self.cap is not None:
            ret, frame = self.cap.read()
            return frame if ret else None
        if self.tello is not None:
            frame = self.tello.get_frame_read().frame
            return frame if frame is not None and frame.size > 0 else None
        return None

    def _close_source(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.tello is not None:
            try:
                self.tello.streamoff()
                self.tello.end()
            except Exception:  # pragma: no cover
                pass
            self.tello = None

    # ---------------------------------------------------------------------
    # Prediction loop -----------------------------------------------------
    # ---------------------------------------------------------------------

    def _loop(self) -> None:  # noqa: D401 – simple descriptor
        if not self._open_source():
            print("Could not open video source – detector aborting")
            self.running = False
            return
        try:
            while self.running:
                frame = self._read_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                self.last_frame = frame
                with torch.no_grad():
                    tensor = preprocess_image(frame).to(self.device)
                    outputs = self.model(tensor)
                    _, pred = torch.max(outputs, 1)
                    self.last_prediction = CLASSES[pred.item()]
                # Keep FPS reasonable when using CPU
                time.sleep(0.05)
        finally:
            self._close_source()

    # ---------------------------------------------------------------------
    # Public control methods ---------------------------------------------
    # ---------------------------------------------------------------------

    def start(self) -> bool:
        if self.running:
            return True
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
        self._close_source()


# ---------------------------------------------------------------------------
# FastAPI --------------------------------------------------------------------
# ---------------------------------------------------------------------------

app = FastAPI(title="Drone Plant Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One global detector instance (created lazily on first /start call)
_detector: Optional[Detector] = None


def get_detector(source: str = "webcam") -> Detector:  # noqa: D401 – simple descriptor
    global _detector  # noqa: PLW0603
    if _detector is None or _detector.source != source:
        if _detector is not None:
            _detector.stop()
        _detector = Detector(source=source)
    return _detector


@app.get("/start")
def start_detection(source: str = Query("tello", enum=["tello", "webcam"])) -> JSONResponse:  # type: ignore[valid-type]
    det = get_detector(source)
    det.start()
    return JSONResponse({"status": "running", "source": source})


@app.get("/stop")
def stop_detection() -> JSONResponse:  # noqa: D401 – simple descriptor
    if _detector is not None:
        _detector.stop()
    return JSONResponse({"status": "stopped"})


@app.get("/latest_prediction")
def latest_prediction() -> JSONResponse:  # noqa: D401 – simple descriptor
    if _detector is None or _detector.last_prediction is None:
        return JSONResponse({"prediction": None})
    return JSONResponse({"prediction": _detector.last_prediction})


@app.get("/latest_frame")
def latest_frame() -> Response:  # noqa: D401 – simple descriptor
    if _detector is None or _detector.last_frame is None:
        return Response(content=b"", media_type="image/jpeg", status_code=204)
    # Encode as JPEG
    success, buf = cv2.imencode(".jpg", _detector.last_frame)
    if not success:
        return Response(content=b"", media_type="image/jpeg", status_code=500)
    return Response(content=buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/")
def root() -> JSONResponse:  # noqa: D401 – simple descriptor
    """Simple health check."""
    return JSONResponse({"detail": "Drone Detection API is up"})
