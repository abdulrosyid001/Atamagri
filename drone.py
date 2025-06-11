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

# Define the necessary model architecture classes
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
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Define the class names (from your training data)
classes = ['Tomato__Late_blight', 'Tomato_healthy', 'Grape_healthy', 'Orange_Haunglongbing(Citrus_greening)', 
          'Soybean__healthy', 'Squash_Powdery_mildew', 'Potato_healthy', 'Corn(maize)___Northern_Leaf_Blight', 
          'Tomato__Early_blight', 'Tomato_Septoria_leaf_spot', 'Corn(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
          'Strawberry__Leaf_scorch', 'Peach_healthy', 'Apple_Apple_scab', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 
          'Tomato__Bacterial_spot', 'Apple_Black_rot', 'Blueberry_healthy', 'Cherry(including_sour)___Powdery_mildew', 
          'Peach__Bacterial_spot', 'Apple_Cedar_apple_rust', 'Tomato_Target_Spot', 'Pepper,_bell__healthy', 
          'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Potato__Late_blight', 'Tomato__Tomato_mosaic_virus', 
          'Strawberry__healthy', 'Apple_healthy', 'Grape_Black_rot', 'Potato__Early_blight', 
          'Cherry_(including_sour)__healthy', 'Corn(maize)__Common_rust', 'Grape__Esca(Black_Measles)', 
          'Raspberry__healthy', 'Tomato_Leaf_Mold', 'Tomato__Spider_mites Two-spotted_spider_mite', 
          'Pepper,bell_Bacterial_spot', 'Corn(maize)___healthy']

# Color calibration constants and settings
class ColorCalibration:
    def __init__(self):
        # Auto calibration settings
        self.is_auto_calibration = True
        self.calibration_frames = 30  # Number of frames to use for auto calibration
        self.frames_collected = 0
        self.calibration_complete = False
        self.reference_values = None
        self.color_correction_matrix = None
        self.white_balance_gains = [1.0, 1.0, 1.0]  # Default BGR gains
        
        # Manual calibration settings
        self.red_gain = 1.0
        self.green_gain = 1.0
        self.blue_gain = 1.0
        self.brightness = 0
        self.contrast = 1.0
        
        # Settings specifically for purple tint correction
        self.anti_purple_strength = 0.3  # Default strength for purple correction
    
    def init_calibration(self):
        """Reset calibration to start over"""
        self.frames_collected = 0
        self.calibration_complete = False
        self.reference_values = None
        self.color_correction_matrix = None
        
    def auto_white_balance(self, frame):
        """Apply automatic white balance using Gray World assumption"""
        if frame is None or frame.size == 0:
            return frame
            
        # Split the channels
        b, g, r = cv2.split(frame.astype(np.float32))
        
        # Calculate average of each channel
        b_avg = np.mean(b)
        g_avg = np.mean(g)
        r_avg = np.mean(r)
        
        # Calculate average intensity
        intensity_avg = (b_avg + g_avg + r_avg) / 3
        
        # Calculate scaling factors
        if b_avg > 0 and g_avg > 0 and r_avg > 0:
            kb = intensity_avg / b_avg
            kg = intensity_avg / g_avg
            kr = intensity_avg / r_avg
            
            # Apply scaling with some dampening to prevent overcorrection
            dampening = 0.7
            self.white_balance_gains = [
                (1.0 - dampening) * self.white_balance_gains[0] + dampening * kb,
                (1.0 - dampening) * self.white_balance_gains[1] + dampening * kg,
                (1.0 - dampening) * self.white_balance_gains[2] + dampening * kr
            ]
            
            # Apply gains
            b = np.clip(b * self.white_balance_gains[0], 0, 255)
            g = np.clip(g * self.white_balance_gains[1], 0, 255)
            r = np.clip(r * self.white_balance_gains[2], 0, 255)
            
            # Merge channels back
            return cv2.merge([b, g, r]).astype(np.uint8)
        
        return frame
    
    def correct_purple_tint(self, frame):
        """Specific correction for purple tint - reduces blue channel in areas with high blue and red"""
        if frame is None or frame.size == 0:
            return frame
            
        # Convert to float for calculations
        frame_float = frame.astype(np.float32)
        
        # Split channels
        b, g, r = cv2.split(frame_float)
        
        # Identify purple areas (high blue and red, lower green)
        purple_mask = ((b > r * 0.8) & (b > g * 1.2) & (r > g * 1.2)).astype(np.float32)
        
        # Reduce blue channel in purple areas
        adjustment = purple_mask * self.anti_purple_strength * b
        b = np.clip(b - adjustment, 0, 255)
        
        # Slightly boost green in purple areas to neutralize
        g = np.clip(g + (adjustment * 0.5), 0, 255)
        
        # Merge channels
        result = cv2.merge([b, g, r]).astype(np.uint8)
        return result
    
    def apply_manual_calibration(self, frame):
        """Apply manual RGB gains, brightness and contrast"""
        if frame is None or frame.size == 0:
            return frame
            
        # Split channels
        b, g, r = cv2.split(frame.astype(np.float32))
        
        # Apply channel gains
        b = np.clip(b * self.blue_gain, 0, 255)
        g = np.clip(g * self.green_gain, 0, 255)
        r = np.clip(r * self.red_gain, 0, 255)
        
        # Merge channels
        adjusted = cv2.merge([b, g, r])
        
        # Apply brightness and contrast
        adjusted = cv2.convertScaleAbs(adjusted, alpha=self.contrast, beta=self.brightness)
        
        return adjusted
    
    def process_frame(self, frame):
        if frame is None or frame.size == 0:
            return frame
        
        # Convert to 3 channels if necessary
        if frame.ndim == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # BGRA
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Apply auto white balance if enabled
        if self.is_auto_calibration:
            frame = self.auto_white_balance(frame)
        
        # Apply purple tint correction
        frame = self.correct_purple_tint(frame)
        
        # Apply manual calibration on top
        frame = self.apply_manual_calibration(frame)
        
        # During calibration phase, collect frames
        if self.is_auto_calibration and not self.calibration_complete and self.frames_collected < self.calibration_frames:
            self.frames_collected += 1
            if self.frames_collected >= self.calibration_frames:
                self.calibration_complete = True
                print("Automatic color calibration complete")
        
        return frame

# Function to preprocess image for the model
def preprocess_image(image):
    # Convert BGR to RGB (OpenCV uses BGR, PyTorch uses RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 256x256
    image = cv2.resize(image, (256, 256))
    
    # Convert to PIL Image
    image = Image.fromarray(image)
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    return transform(image)

# Function to predict disease from an image
def predict_disease(img, model):
    # Preprocess the image
    img_tensor = preprocess_image(img)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    # Move to same device as model
    img_tensor = img_tensor.to(next(model.parameters()).device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        
    return classes[predicted.item()]

def initialize_tello(retry_count=3):
    """Initialize the Tello drone and prepare for video streaming with error handling"""
    for attempt in range(retry_count):
        try:
            tello = Tello()
            
            # Connect to the drone
            tello.connect()
            print("Connected to Tello drone")
            
            # Check battery level
            battery_level = tello.get_battery()
            print(f"Battery Level: {battery_level}%")
            
            if battery_level < 20:
                print(f"WARNING: Battery level is low ({battery_level}%). Flying may be restricted.")
            
            # Put drone in command mode and start video stream
            tello.streamon()
            print("Stream started")
            
            # Get a test frame to verify stream is working
            for _ in range(5):  # Try a few times to get a frame
                frame = tello.get_frame_read().frame
                if frame is not None and frame.size > 0:
                    print("Successfully received video frame from Tello")
                    return tello, True  # Return drone object and flight capability flag
                time.sleep(0.5)
            
            print("WARNING: Connected to Tello but video stream is not providing frames")
            return tello, False  # Return drone but mark as not flight capable
            
        except Exception as e:
            print(f"Attempt {attempt+1}/{retry_count} failed: {str(e)}")
            if attempt < retry_count - 1:
                print("Retrying connection to Tello...")
                time.sleep(2)  # Wait before retrying
            else:
                print("Failed to initialize Tello after multiple attempts")
                return None, False
    
    return None, False

def get_safe_tello_frame(tello, default_frame):
    """Safely get a frame from Tello with error handling"""
    try:
        frame = tello.get_frame_read().frame
        if frame is not None and frame.size > 0:
            return frame
    except Exception as e:
        print(f"Error getting Tello frame: {str(e)}")
    
    return default_frame  # Return the default frame if we couldn't get one from Tello

def safe_tello_command(tello, command_func, *args, **kwargs):
    """Execute a Tello command with error handling"""
    if tello is None:
        return False
    
    try:
        command_func(*args, **kwargs)
        return True
    except Exception as e:
        print(f"Error executing Tello command: {str(e)}")
        return False

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize color calibration
    calibration = ColorCalibration()
    print("Color calibration initialized - auto-correcting for purple tint")
    
    # Load the model - METHOD 1: Using weights_only=False (less secure but works)
    try:
        model_path = "plant-disease-model-complete (1).pth"
        print("Attempting to load model with weights_only=False...")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model with weights_only=False: {e}")
        
        # METHOD 2: Try loading with safe_globals
        try:
            print("Attempting to load model with safe_globals...")
            from torch.serialization import safe_globals
            with safe_globals([ResNet9]):
                model = torch.load(model_path, map_location=device)
            model.eval()
            print("Model loaded successfully with safe_globals!")
        except Exception as e:
            print(f"Error loading model with safe_globals: {e}")
            
            # METHOD 3: Try loading just the state dict
            try:
                print("Attempting to create a new model and load state dict...")
                # Create a new instance of the model
                model = ResNet9(3, len(classes))
                model.to(device)
                
                # Load just the state dict
                state_dict = torch.load(model_path, map_location=device)
                # Check if what we loaded is already a state dict
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    model.load_state_dict(state_dict['state_dict'])
                elif isinstance(state_dict, nn.Module):
                    # If we loaded a full model, get its state dict
                    model.load_state_dict(state_dict.state_dict())
                else:
                    # Otherwise assume it's a direct state dict
                    model.load_state_dict(state_dict)
                    
                model.eval()
                print("Model loaded successfully with state dict approach!")
            except Exception as e:
                print(f"Error loading model state dict: {e}")
                print("Could not load the model. Please check the model file.")
                return
    
    # Ask user if they want to use drone or webcam
    camera_source = input("Select camera source ('tello' for drone, 'webcam' for webcam, 'both' to use both): ").lower()
    
    # Initialize webcam if needed
    cap = None
    if camera_source in ['webcam', 'both']:
        camera_id = 0  # Default webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Could not open webcam with ID {camera_id}. Trying external camera...")
            cap = cv2.VideoCapture(1)  # Try external camera
            
        if not cap.isOpened() and camera_source == 'webcam':
            print("Error: Could not open any camera.")
            return
        elif not cap.isOpened() and camera_source == 'both':
            print("Webcam not available. Using only the Tello drone camera.")
            camera_source = 'tello'
        
        # Set frame dimensions (optional, depends on your camera)
        if cap and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize Tello if needed
    tello = None
    flight_capable = False
    if camera_source in ['tello', 'both']:
        try:
            print("Initializing Tello drone...")
            tello, flight_capable = initialize_tello()
            
            if tello is None:
                print("Could not initialize Tello drone.")
                if camera_source == 'tello':
                    print("Falling back to webcam...")
                    camera_source = 'webcam'
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        print("Error: Could not open webcam either.")
                        return
                elif camera_source == 'both':
                    print("Using only webcam.")
                    camera_source = 'webcam'
            else:
                print(f"Tello drone initialized successfully! Flight capable: {flight_capable}")
                if not flight_capable:
                    print("WARNING: Drone video stream is available, but takeoff capability is disabled due to errors")
        except Exception as e:
            print(f"Error during Tello initialization: {e}")
            if camera_source == 'tello':
                print("Falling back to webcam...")
                camera_source = 'webcam'
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Error: Could not open webcam either.")
                    return
            elif camera_source == 'both':
                print("Using only webcam.")
                camera_source = 'webcam'
    
    print("\nInstructions:")
    print("Press 'q' to quit")
    print("Press 'c' to switch between webcam sources (if available)")
    print("Press 't' to switch between drone and webcam (if both available)")
    # Color calibration controls
    print("\nColor calibration controls:")
    print("  'a' - toggle auto calibration on/off")
    print("  'r' - reset calibration")
    print("  'p' - reduce purple tint (+ to increase, - to decrease)")
    print("  '1/2' - decrease/increase red gain")
    print("  '3/4' - decrease/increase green gain")
    print("  '5/6' - decrease/increase blue gain")
    print("  '7/8' - decrease/increase brightness")
    print("  '9/0' - decrease/increase contrast")
    
    if tello and flight_capable:
        print("\nDrone controls (enabled):")
        print("  'w' - move forward")
        print("  's' - move backward")
        print("  'a' - move left")
        print("  'd' - move right")
        print("  'up arrow' - move up")
        print("  'down arrow' - move down")
        print("  'left arrow' - rotate counter-clockwise")
        print("  'right arrow' - rotate clockwise")
        print("  'space' - takeoff/land toggle")
    elif tello and not flight_capable:
        print("Drone video available, but flight controls are disabled due to errors")
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Track current camera source
    current_source = camera_source
    if camera_source == 'both':
        current_source = 'tello' if tello else 'webcam'
    
    # Track Tello flying state
    is_flying = False
    
    # Default frame for when Tello frame retrieval fails
    default_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(default_frame, "No video signal", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show calibration status
    show_calibration_info = True
    
    # Main loop
    while True:
        # FPS calculation
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        # Update FPS every second
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Get frame based on current source
        frame_available = True
        if current_source == 'tello' and tello:
            try:
                frame = get_safe_tello_frame(tello, default_frame)
                source_text = "Source: Tello Drone"
            except Exception as e:
                print(f"Error getting Tello frame: {e}")
                frame = default_frame
                source_text = "Source: Tello (Error)"
        elif cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam")
                if tello and camera_source == 'both':
                    print("Switching to Tello camera")
                    current_source = 'tello'
                    continue
                else:
                    frame_available = False
                    break
            source_text = "Source: Webcam"
        else:
            print("No video source available")
            break
        
        if frame_available:
            try:
                # Store the original frame for comparison if needed
                original_frame = frame.copy()
                
                # Apply color calibration
                frame = calibration.process_frame(frame)
                
                # Get prediction
                prediction = predict_disease(frame, model)
                
                # Prepare result info
                plant, condition = prediction.split('_', 1)
                
                # Display result on frame
                cv2.putText(frame, source_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Plant: {plant}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Condition: {condition}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display FPS
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display battery level if using Tello
                if current_source == 'tello' and tello:
                    try:
                        battery = tello.get_battery()
                        cv2.putText(frame, f"Battery: {battery}%", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, f"Flight capable: {flight_capable}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    except:
                        cv2.putText(frame, "Battery: Unknown", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display calibration info if enabled
                if show_calibration_info:
                    y_pos = 270
                    cv2.putText(frame, f"Auto Calibration: {'ON' if calibration.is_auto_calibration else 'OFF'}", 
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_pos += 30
                    cv2.putText(frame, f"Anti-Purple: {calibration.anti_purple_strength:.2f}", 
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_pos += 30
                    cv2.putText(frame, f"RGB Gains: {calibration.red_gain:.2f}, {calibration.green_gain:.2f}, {calibration.blue_gain:.2f}", 
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_pos += 30
                    cv2.putText(frame, f"Brightness: {calibration.brightness}, Contrast: {calibration.contrast:.2f}", 
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show the frame
                cv2.imshow('Plant Disease Detection', frame)
                
                # Optional: Show original frame for comparison
                # cv2.imshow('Original Frame', original_frame)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                # Show error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Error processing frame", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow('Plant Disease Detection', error_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Quit on 'q' key
        if key == ord('q'):
            break
            
        # Switch camera on 'c' key (webcam only)
        elif key == ord('c') and cap and cap.isOpened() and current_source == 'webcam':
            try:
                current_camera = int(cap.get(cv2.CAP_PROP_INDEX))
                next_camera = (current_camera + 1) % 2  # Toggle between 0 and 1
                
                # Release current camera and try to open the next one
                cap.release()
                cap = cv2.VideoCapture(next_camera)
                
                if not cap.isOpened():
                    print(f"Could not open camera with ID {next_camera}, reverting to previous camera")
                    cap = cv2.VideoCapture(current_camera)
                else:
                    print(f"Switched to webcam {next_camera}")
            except Exception as e:
                print(f"Error switching webcam: {e}")
                
        # Switch between Tello and webcam on 't' key
        elif key == ord('t') and camera_source == 'both':
            if current_source == 'tello' and cap and cap.isOpened():
                current_source = 'webcam'
                print("Switched to webcam")
            elif current_source == 'webcam' and tello:
                current_source = 'tello'
                print("Switched to Tello camera")
        
        # Toggle auto calibration
        elif key == ord('a'):
            calibration.is_auto_calibration = not calibration.is_auto_calibration
            print(f"Auto calibration: {'ON' if calibration.is_auto_calibration else 'OFF'}")
            
        # Reset calibration
        elif key == ord('r'):
            calibration.init_calibration()
            calibration.red_gain = 1.0
            calibration.green_gain = 1.0
            calibration.blue_gain = 1.0
            calibration.brightness = 0
            calibration.contrast = 1.0
            calibration.anti_purple_strength = 0.3
            calibration.white_balance_gains = [1.0, 1.0, 1.0]
            print("Calibration reset to defaults")
            
        # Toggle calibration info display
        elif key == ord('i'):
            show_calibration_info = not show_calibration_info
            
        # Adjust anti-purple strength
        elif key == ord('p'):
            calibration.anti_purple_strength = min(1.0, calibration.anti_purple_strength + 0.05)
            print(f"Anti-purple strength: {calibration.anti_purple_strength:.2f}")
        elif key == ord('-'):
            calibration.anti_purple_strength = max(0.0, calibration.anti_purple_strength - 0.05)
            print(f"Anti-purple strength: {calibration.anti_purple_strength:.2f}")
        elif key == ord('+'):
            calibration.anti_purple_strength = min(1.0, calibration.anti_purple_strength + 0.05)
            print(f"Anti-purple strength: {calibration.anti_purple_strength:.2f}")
            
        # Adjust red gain
        elif key == ord('1'):
            calibration.red_gain = max(0.1, calibration.red_gain - 0.05)
            print(f"Red gain: {calibration.red_gain:.2f}")
        elif key == ord('2'):
            calibration.red_gain = min(3.0, calibration.red_gain + 0.05)
            print(f"Red gain: {calibration.red_gain:.2f}")
            
        # Adjust green gain
        elif key == ord('3'):
            calibration.green_gain = max(0.1, calibration.green_gain - 0.05)
            print(f"Green gain: {calibration.green_gain:.2f}")
        elif key == ord('4'):
            calibration.green_gain = min(3.0, calibration.green_gain + 0.05)
            print(f"Green gain: {calibration.green_gain:.2f}")
            
        # Adjust blue gain
        elif key == ord('5'):
            calibration.blue_gain = max(0.1, calibration.blue_gain - 0.05)
            print(f"Blue gain: {calibration.blue_gain:.2f}")
        elif key == ord('6'):
            calibration.blue_gain = min(3.0, calibration.blue_gain + 0.05)
            print(f"Blue gain: {calibration.blue_gain:.2f}")
            
        # Adjust brightness
        elif key == ord('7'):
            calibration.brightness = max(-50, calibration.brightness - 5)
            print(f"Brightness: {calibration.brightness}")
        elif key == ord('8'):
            calibration.brightness = min(50, calibration.brightness + 5)
            print(f"Brightness: {calibration.brightness}")
            
        # Adjust contrast
        elif key == ord('9'):
            calibration.contrast = max(0.5, calibration.contrast - 0.1)
            print(f"Contrast: {calibration.contrast:.2f}")
        elif key == ord('0'):
            calibration.contrast = min(2.0, calibration.contrast + 0.1)
            print(f"Contrast: {calibration.contrast:.2f}")
        
        # Drone control keys - only if flight_capable is True
        if tello and current_source == 'tello' and flight_capable:
            # Takeoff/land toggle
            if key == ord(' '):  # Space bar
                if not is_flying:
                    print("Attempting takeoff...")
                    if safe_tello_command(tello, tello.takeoff):
                        is_flying = True
                        print("Tello took off successfully")
                    else:
                        print("Takeoff failed!")
                else:
                    print("Landing...")
                    if safe_tello_command(tello, tello.land):
                        is_flying = False
                        print("Tello landed successfully")
                    else:
                        print("Landing command failed!")
            
            # Only process movement commands if drone is flying
            if is_flying:
                # Forward/backward
                if key == ord('w'):
                    safe_tello_command(tello, tello.move_forward, 30)
                elif key == ord('s'):
                    safe_tello_command(tello, tello.move_back, 30)
                
                # Left/right
                if key == ord('a'):
                    safe_tello_command(tello, tello.move_left, 30)
                elif key == ord('d'):
                    safe_tello_command(tello, tello.move_right, 30)
                
                # Up/down
                if key == 82:  # Up arrow
                    safe_tello_command(tello, tello.move_up, 30)
                elif key == 84:  # Down arrow
                    safe_tello_command(tello, tello.move_down, 30)
                
                # Rotate
                if key == 81:  # Left arrow
                    safe_tello_command(tello, tello.rotate_counter_clockwise, 30)
                elif key == 83:  # Right arrow
                    safe_tello_command(tello, tello.rotate_clockwise, 30)
    
    # Clean up and release resources
    if cap:
        cap.release()
    
    if tello:
        try:
            print("Shutting down Tello connection...")
            if is_flying:
                try:
                    tello.land()
                    print("Emergency landing initiated")
                    time.sleep(3)  # Wait for landing to complete
                except:
                    print("Emergency landing failed")
            
            try:
                tello.streamoff()
                print("Video stream ended")
            except:
                print("Error ending video stream")
                
            try:
                tello.end()
                print("Tello connection closed")
            except:
                print("Error closing Tello connection")
        except Exception as e:
            print(f"Error during Tello shutdown: {e}")
    
    cv2.destroyAllWindows()
    print("Application terminated successfully")

if __name__ == "__main__":
    main()