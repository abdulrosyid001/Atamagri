import os
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import time
from djitellopy import Tello
import threading
import asyncio
import websockets
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Define the ResNet9 model architecture that matches your saved model
class ResNet9(nn.Module):
    def __init__(self, num_classes):
        super(ResNet9, self).__init__()
        
        # First convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # First residual block
        self.res1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        )
        
        # Third convolution block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Fourth convolution block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Second residual block
        self.res2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        # First residual connection
        residual = x
        x = self.res1(x)
        x = x + residual
        
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Second residual connection
        residual = x
        x = self.res2(x)
        x = x + residual
        
        x = self.classifier(x)
        return x

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
tello = None
webcam = None
is_streaming = False
stream_thread = None
model = None
calibration = None
use_webcam = False
current_mode = "disconnected"  # "tello", "webcam", "disconnected"

# Define class names - Complete PlantVillage dataset (38 classes)
classes = [
    "Apple_Apple_scab",
    "Apple_Black_rot", 
    "Apple_Cedar_apple_rust",
    "Apple_healthy",
    "Blueberry_healthy",
    "Cherry_Powdery_mildew",
    "Cherry_healthy",
    "Corn_Common_rust",
    "Corn_Gray_leaf_spot",
    "Corn_Leaf_blight",
    "Corn_healthy",
    "Grape_Black_rot",
    "Grape_Esca",
    "Grape_Leaf_blight",
    "Grape_healthy",
    "Orange_Huanglongbing",
    "Peach_Bacterial_spot",
    "Peach_healthy",
    "Pepper_bell_Bacterial_spot",
    "Pepper_bell_healthy",
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Potato_healthy",
    "Raspberry_healthy",
    "Soybean_healthy",
    "Squash_Powdery_mildew",
    "Strawberry_Leaf_scorch",
    "Strawberry_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

# Load the model
def load_model():
    global model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model_path = "model/plant-disease-model-complete (1).pth"
        
        # Try to load the complete model first
        try:
            model = torch.load(model_path, map_location=device, weights_only=False)
            print("Loaded complete model successfully!")
        except Exception as e:
            print(f"Failed to load complete model: {e}")
            # Fallback: Initialize model and load state dict
            model = ResNet9(num_classes=len(classes))
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint.state_dict())
            print("Loaded model from state dict!")
        
        model = model.to(device)
        model.eval()
        print("Model is ready for inference!")
        
        # Test the model with a dummy input
        try:
            print("Testing model with dummy input...")
            dummy_input = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                test_output = model(dummy_input)
                print(f"Model test successful! Output shape: {test_output.shape}")
                print(f"Number of classes: {test_output.shape[1]}")
                print(f"Available classes in list: {len(classes)}")
                
                if test_output.shape[1] != len(classes):
                    print(f"WARNING: Model outputs {test_output.shape[1]} classes but we have {len(classes)} class names!")
                    
        except Exception as test_error:
            print(f"Model test failed: {test_error}")
            
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Initialize drone or webcam
def init_drone():
    global tello, webcam, use_webcam, current_mode
    
    # First try to connect to Tello drone
    try:
        print("Attempting to connect to Tello drone...")
        tello = Tello()
        tello.connect()
        print("Connected to Tello drone")
        
        battery = tello.get_battery()
        print(f"Battery Level: {battery}%")
        
        if battery < 20:
            print(f"WARNING: Battery level is low ({battery}%). Flying may be restricted.")
        
        tello.streamon()
        print("Tello stream started")
        use_webcam = False
        current_mode = "tello"
        return True
    except Exception as e:
        print(f"Error connecting to Tello drone: {e}")
        print("Falling back to webcam...")
        
        # Fallback to webcam
        try:
            import cv2
            print("Trying to initialize webcam...")
            
            webcam = cv2.VideoCapture(0)  # Try default camera
            if not webcam.isOpened():
                print("Default camera (0) failed, trying other indices...")
                # Try other camera indices
                for i in range(1, 4):
                    print(f"Trying camera index {i}...")
                    webcam = cv2.VideoCapture(i)
                    if webcam.isOpened():
                        print(f"Camera {i} opened successfully")
                        break
                else:
                    raise Exception("No webcam found")
            else:
                print("Default camera (0) opened successfully")
            
            # Set webcam properties for better performance
            webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            webcam.set(cv2.CAP_PROP_FPS, 30)
            
            # Test if we can read from webcam
            print("Testing webcam capture...")
            ret, frame = webcam.read()
            if not ret or frame is None:
                raise Exception("Cannot read from webcam")
            
            print(f"Webcam test successful! Frame shape: {frame.shape}")
            use_webcam = True
            current_mode = "webcam"
            return True
        except Exception as webcam_error:
            print(f"Error initializing webcam: {webcam_error}")
            current_mode = "disconnected"
            return False

# Process frame for disease detection
def process_frame(frame):
    if frame is None or frame.size == 0:
        return None
    
    # Skip processing every few frames for performance
    import random
    if random.random() > 0.3:  # Process only 30% of frames
        return None
    
    try:
        print("Processing frame for disease detection...")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"Frame converted to RGB: {frame_rgb.shape}")
        
        # Resize to 256x256
        frame_resized = cv2.resize(frame_rgb, (256, 256))
        print(f"Frame resized to: {frame_resized.shape}")
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_resized)
        
        # Apply transformations (try without normalization first)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Transform image
        img_tensor = transform(pil_image)
        img_tensor = img_tensor.unsqueeze(0)
        print(f"Tensor shape: {img_tensor.shape}")
        
        # Get prediction
        with torch.no_grad():
            print("Running model inference...")
            outputs = model(img_tensor)
            print(f"Model output shape: {outputs.shape}")
            print(f"Model output values: {outputs}")
            
            _, predicted = torch.max(outputs, 1)
            confidence = F.softmax(outputs, dim=1)[0][predicted].item()
            
            print(f"Predicted class index: {predicted.item()}")
            print(f"Confidence: {confidence}")
            
        # Get class index and validate
        predicted_idx = predicted.item()
        if predicted_idx >= len(classes):
            print(f"Warning: Predicted index {predicted_idx} out of range for {len(classes)} classes")
            return None
            
        # Get class name
        prediction = classes[predicted_idx]
        print(f"Predicted class: {prediction}")
        
        # Split plant and condition safely
        if '_' in prediction:
            parts = prediction.split('_', 1)
            plant = parts[0]
            condition = parts[1]
        else:
            plant = prediction
            condition = "Unknown"
        
        result = {
            "plant": plant,
            "condition": condition,
            "confidence": round(confidence * 100, 2)
        }
        print(f"Final prediction result: {result}")
        return result
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return None

# WebSocket connection handler
async def websocket_handler(websocket: WebSocket):
    await websocket.accept()
    print(f"Client connected - Current mode: {current_mode}")
    
    frame_count = 0
    try:
        while True:
            frame = None
            
            # Get frame from appropriate source
            if current_mode == "tello" and tello:
                frame = tello.get_frame_read().frame
                if frame is not None:
                    print(f"Got Tello frame: {frame.shape}")
            elif current_mode == "webcam" and webcam:
                ret, frame = webcam.read()
                if not ret:
                    print("Failed to read from webcam")
                    continue
                if frame is not None:
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30 frames
                        print(f"Got webcam frame {frame_count}: {frame.shape}")
            
            if frame is not None:
                try:
                    # Process frame for disease detection
                    prediction = process_frame(frame)
                    
                    # Convert frame to JPEG
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = buffer.tobytes()
                    
                    # Send frame
                    await websocket.send_bytes(frame_bytes)
                    
                    # Send prediction if available
                    if prediction:
                        await websocket.send_text(json.dumps(prediction))
                        
                except Exception as frame_error:
                    print(f"Error processing frame: {frame_error}")
            else:
                if current_mode == "webcam":
                    print("No frame captured from webcam")
            
            await asyncio.sleep(0.1)  # 10 FPS
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("Client disconnected")

# FastAPI routes
@app.get("/")
async def root():
    return {"status": "Drone server is running"}

@app.get("/connect")
async def connect():
    if init_drone():
        if current_mode == "tello":
            battery = tello.get_battery() if tello else 0
            return {
                "success": True, 
                "message": "Connected to Tello drone",
                "mode": "tello",
                "battery": battery
            }
        elif current_mode == "webcam":
            return {
                "success": True, 
                "message": "Connected to webcam (drone simulation mode)",
                "mode": "webcam",
                "battery": 100  # Simulated battery
            }
    return {"success": False, "message": "Failed to connect to any device", "mode": "disconnected"}

class CommandRequest(BaseModel):
    command: str

@app.post("/command")
async def command(request: CommandRequest):
    if current_mode == "disconnected":
        return {"success": False, "message": "No device connected"}
    
    try:
        if current_mode == "tello" and tello:
            # Execute real Tello commands
            if request.command == "takeoff":
                tello.takeoff()
            elif request.command == "land":
                tello.land()
            elif request.command == "emergency":
                tello.emergency()
            elif request.command == "up":
                tello.move_up(30)
            elif request.command == "down":
                tello.move_down(30)
            elif request.command == "left":
                tello.move_left(30)
            elif request.command == "right":
                tello.move_right(30)
            elif request.command == "forward":
                tello.move_forward(30)
            elif request.command == "back":
                tello.move_back(30)
            elif request.command == "flip":
                tello.flip("f")
            elif request.command == "rotate_cw":
                tello.rotate_clockwise(30)
            elif request.command == "rotate_ccw":
                tello.rotate_counter_clockwise(30)
            else:
                return {"success": False, "message": "Invalid command"}
            
            return {"success": True, "message": f"Tello command {request.command} executed successfully"}
            
        elif current_mode == "webcam":
            # Simulate drone commands for webcam mode
            valid_commands = ["takeoff", "land", "emergency", "up", "down", "left", "right", 
                            "forward", "back", "flip", "rotate_cw", "rotate_ccw"]
            
            if request.command in valid_commands:
                print(f"Simulated command: {request.command}")
                return {"success": True, "message": f"Simulated command {request.command} executed (webcam mode)"}
            else:
                return {"success": False, "message": "Invalid command"}
        
    except Exception as e:
        return {"success": False, "message": f"Error executing command: {str(e)}"}

@app.get("/status")
async def get_status():
    """Get current device status and mode"""
    if current_mode == "tello" and tello:
        try:
            battery = tello.get_battery()
            return {
                "mode": "tello",
                "connected": True,
                "battery": battery,
                "signal": 100,  # Assume good signal if connected
                "message": "Tello drone connected"
            }
        except:
            return {
                "mode": "disconnected",
                "connected": False,
                "battery": 0,
                "signal": 0,
                "message": "Tello connection lost"
            }
    elif current_mode == "webcam" and webcam:
        return {
            "mode": "webcam",
            "connected": True,
            "battery": 100,  # Simulated
            "signal": 100,   # Simulated
            "message": "Webcam connected (simulation mode)"
        }
    else:
        return {
            "mode": "disconnected",
            "connected": False,
            "battery": 0,
            "signal": 0,
            "message": "No device connected"
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_handler(websocket)

# Cleanup function
def cleanup():
    global webcam, tello
    if webcam:
        webcam.release()
        print("Webcam released")
    if tello:
        try:
            tello.streamoff()
        except:
            pass
        print("Tello disconnected")

# Event handlers
@app.on_event("shutdown")
async def shutdown_event():
    cleanup()

if __name__ == "__main__":
    # Load model
    if not load_model():
        print("Failed to load model. Exiting...")
        exit(1)
    
    try:
        # Start FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup() 