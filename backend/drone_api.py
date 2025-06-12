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
from typing import Optional, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from fastapi import FastAPI, Query, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from PIL import Image
import sys
import asyncio
from queue import Queue
import base64
import traceback

try:
    from djitellopy import Tello  # Optional – only required for drone mode
except ImportError:  # pragma: no cover
    Tello = None  # type: ignore  # noqa: N816

# ---------------------------------------------------------------------------
# Model setup ----------------------------------------------------------------
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "model" / "plant-disease-model-complete (1).pth"
if not MODEL_PATH.exists():
    # fallback path in case space gets replaced or moved
    MODEL_PATH = Path(__file__).parent / "model" / "plant-disease-model-complete.pth"

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
    "Rice_healthy",
    "Rice_Bacterial_leaf_blight",
    "Rice_Brown_spot",
    "Rice_Leaf_smut",
    "Wheat_healthy",
    "Wheat_Leaf_rust",
    "Wheat_Stem_rust",
    "Wheat_Yellow_rust",
    "Cotton_healthy",
    "Cotton_Bacterial_blight",
    "Cotton_Leaf_curl_virus",
    "Cotton_Leaf_spot"
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


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))  # Keep 2x2 pooling
    return nn.Sequential(*layers)


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        # Initial convolution
        self.conv1 = ConvBlock(in_channels, 64)
        
        # First residual block
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        # Second residual block
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        # Classifier with adaptive pooling
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # This will handle any input size
            nn.Flatten(),
            nn.Linear(512, num_diseases)
        )
        
    def forward(self, xb):
        # Initial convolution
        out = self.conv1(xb)
        
        # First residual block
        out = self.conv2(out)
        out = self.res1(out) + out
        
        # Second residual block
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        
        # Classification
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


def preprocess_image(frame, device):
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to a larger size first
    frame_resized = cv2.resize(frame_rgb, (256, 256))
    
    # Convert to tensor and normalize
    frame_tensor = torch.from_numpy(frame_resized).float()
    frame_tensor = frame_tensor.permute(2, 0, 1)  # Change from HWC to CHW
    frame_tensor = frame_tensor / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    frame_tensor = frame_tensor.unsqueeze(0)
    
    # Move to the same device as the model
    frame_tensor = frame_tensor.to(device)
    
    return frame_tensor


# ---------------------------------------------------------------------------
# Detector class -------------------------------------------------------------
# ---------------------------------------------------------------------------


class Detector:
    """Continuously grab frames from source and run predictions in background."""

    def __init__(self, source: str = "webcam") -> None:
        self.source = source  # 'webcam' or 'tello'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Try to initialize Tello if requested, fall back to webcam if it fails
        if source == "tello":
            try:
                self.tello = Tello()
                self.tello.connect()
                self.tello.streamon()
                print("Successfully connected to Tello drone")
            except Exception as e:
                print(f"Failed to connect to Tello drone: {e}")
                print("Falling back to webcam")
                self.source = "webcam"
                self.tello = None
        else:
            self.tello = None

        # Initialize webcam if needed
        if self.source == "webcam":
            print("Initializing webcam...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open webcam")
            print("Webcam opened successfully")
            
            # Set webcam properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test webcam
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to read from webcam")
            print(f"Webcam test successful! Frame shape: {frame.shape}")

        # Load model checkpoint (handles both full model pickles and state_dicts)
        try:
            # Allow ResNet9 global for safe unpickling
            import torch.serialization as _ts
            _ts.add_safe_globals([ResNet9])

            ckpt = torch.load(
                MODEL_PATH,
                map_location=self.device,
                weights_only=False,  # allow full object
            )

            if isinstance(ckpt, ResNet9):
                # File already contains full model
                self.model = ckpt
            else:
                # Assume state-dict‐like structure
                state_dict = ckpt.get("state_dict", ckpt)
                self.model = ResNet9(3, len(CLASSES))
                self.model.load_state_dict(state_dict)

            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully on", self.device)
        except Exception as e:
            print(f"Failed to load model checkpoint: {e}")
            raise

        # Initialize state
        self.running = False
        self.thread = None
        self.last_frame = None
        self.last_prediction = None

    def _read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the current source."""
        try:
            if self.source == "tello" and self.tello:
                frame = self.tello.get_frame_read().frame
                if frame is not None:
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif self.source == "webcam" and self.cap:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return frame
        except Exception as e:
            print(f"Error reading frame: {e}")
        return None

    def start(self) -> None:
        """Start the detection loop in a background thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the detection loop and clean up resources."""
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.source == "tello" and self.tello is not None:
            self.tello.streamoff()
            self.tello.end()
        elif self.source == "webcam":
            self.cap.release()

    def _loop(self) -> None:
        """Main detection loop (runs in background thread)."""
        print("Starting detection loop (thread)")
        try:
            while self.running:
                frame = self._read_frame()
                if frame is None:
                    # No frame captured, wait briefly
                    time.sleep(0.1)
                    continue

                # Store frame
                self.last_frame = frame
                # Debug: print frame shape occasionally
                # print(f"Frame captured: {frame.shape}")

                # Run prediction
                with torch.no_grad():
                    # Preprocess frame
                    frame_tensor = preprocess_image(frame, self.device)

                    # Get prediction
                    outputs = self.model(frame_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)

                    prediction = {
                        "class": CLASSES[predicted.item()],
                        "confidence": confidence.item(),
                    }

                    self.last_prediction = prediction

                # Prediction loop rate ~10 Hz
                time.sleep(0.1)

        except Exception as e:
            print(f"Error in detection loop: {e}")
            traceback.print_exc()
        finally:
            # Cleanup resources
            if self.source == "tello" and self.tello is not None:
                self.tello.streamoff()
            elif self.source == "webcam":
                self.cap.release()


# ---------------------------------------------------------------------------
# FastAPI --------------------------------------------------------------------
# ---------------------------------------------------------------------------

app = FastAPI(title="Drone Plant Disease Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://10.11.3.229:3000"],  # React frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
detector = None
frame_queue = Queue(maxsize=2)
prediction_queue = Queue(maxsize=2)
status_queue = Queue(maxsize=2)

def get_detector(source: str = "webcam") -> Detector:  # noqa: D401 – simple descriptor
    global detector  # noqa: PLW0603
    if detector is None or detector.source != source:
        if detector is not None:
            detector.stop()
        detector = Detector(source=source)
    return detector


@app.get("/start")
async def start_detection(source: str = "webcam"):
    """Start the detection loop with the specified source."""
    global detector
    try:
        if detector is not None:
            detector.stop()
        detector = Detector(source)
        detector.start()
        return {"status": "success", "message": f"Started detection with {source}"}
    except Exception as e:
        print(f"Error starting detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stop")
async def stop_detection():
    """Stop the detection loop."""
    global detector
    if detector is not None:
        detector.stop()
        detector = None
    return {"status": "success", "message": "Stopped detection"}


@app.get("/latest_prediction")
def latest_prediction() -> JSONResponse:  # noqa: D401 – simple descriptor
    if detector is None or detector.last_prediction is None:
        return JSONResponse({"prediction": None})
    return JSONResponse({"prediction": detector.last_prediction})


@app.get("/latest_frame")
def latest_frame() -> Response:  # noqa: D401 – simple descriptor
    if detector is None or detector.last_frame is None:
        return Response(content=b"", media_type="image/jpeg", status_code=204)
    # Encode as JPEG
    success, buf = cv2.imencode(".jpg", detector.last_frame)
    if not success:
        return Response(content=b"", media_type="image/jpeg", status_code=500)
    return Response(content=buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/")
def root() -> JSONResponse:  # noqa: D401 – simple descriptor
    """Simple health check."""
    return JSONResponse({"detail": "Drone Detection API is up"})

def generate_frames():
    """Generate video frames for streaming."""
    print("Starting video feed generation")
    while True:
        try:
            if detector is None:
                print("No detector available, waiting...")
                time.sleep(0.1)
                continue
                
            frame = detector.last_frame
            if frame is None:
                print("No frame available, waiting...")
                time.sleep(0.1)
                continue
                
            print(f"Generating frame: {frame.shape}")
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                print(f"Frame encoded: {len(frame_bytes)} bytes")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            print(f"Error generating frame: {e}")
            traceback.print_exc()
            time.sleep(0.1)

@app.get("/video_feed")
async def video_feed():
    """Stream video frames."""
    print("Video feed endpoint called")
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    print("WebSocket connection accepted")
    
    try:
        while True:
            if detector is None:
                await asyncio.sleep(0.1)
                continue
                
            # Get latest prediction
            prediction = detector.last_prediction
            if prediction:
                print(f"Sending prediction: {prediction}")
                await websocket.send_json({
                    "type": "prediction",
                    "prediction": prediction
                })
            
            # Get latest status
            status = {
                "connected": True,
                "mode": detector.source,
                "battery": 100 if detector.source == "webcam" else detector.tello.get_battery() if detector.tello else 0,
                "signal": 100 if detector.source == "webcam" else detector.tello.get_wifi_signal_quality() if detector.tello else 0
            }
            await websocket.send_json({
                "type": "status",
                "status": status
            })
            
            await asyncio.sleep(0.1)  # 10 Hz update rate
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        traceback.print_exc()
    finally:
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
