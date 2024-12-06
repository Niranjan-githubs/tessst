from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import json
import logger
import os

# Existing main.py code remains the same
# Add these imports at the top
import platform

def find_camera_source():
    """
    Enhanced camera source detection for Windows
    """
    camera_sources = [
        0,  # Default camera index
        1,  # Alternative index
        'http://localhost:8080/video'  # Optional remote source
    ]

    # Windows-specific logging
    if platform.system() == 'Windows':
        import winreg
        try:
            # Additional Windows camera detection logic
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, 
                r"SYSTEM/CurrentControlSet/Services/usbvideo/Enum"
            )
            logger.info(f"Windows USB Video Devices Detected")
        except Exception as e:
            logger.warning(f"Windows camera registry access failed: {e}")

    for source in camera_sources:
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if cap.isOpened():
                logger.info(f"Successfully opened camera source: {source}")
                return cap
            cap.release()
        except Exception as e:
            logger.warning(f"Failed to open camera source {source}: {e}")
    
    # Fallback virtual frame
    virtual_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    logger.warning("Using virtual black frame as camera source")
    return virtual_frame

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to the script's location
face_cascade_path = os.path.join(base_dir, 'models', 'haarcascade_frontalface_default.xml')
model_path = os.path.join(base_dir,'models', 'antispoofing_full_model.h5')

# Load models using these paths
face_cascade = cv2.CascadeClassifier(face_cascade_path)
model = load_model(model_path)
app = FastAPI()

def convert_numpy_to_json_serializable(obj):
    """Convert NumPy types to standard Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def preprocess_image(frame):
    """Preprocesses the image for face detection while preserving original colors."""
    # Critical: Use the original color space of the frame
    # Do NOT convert or modify color channels
    
    # Create grayscale for face detection WITHOUT modifying original frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray

def detect_and_predict(frame):
    """Detects faces and predicts liveness using the model."""
    # Preprocess the image
    original_image, gray_image = preprocess_image(frame)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    # Process each detected face
    predictions = []
    for (x, y, w, h) in faces:
        # Extract face ROI from ORIGINAL color image
        face_roi = original_image[y:y + h, x:x + w]
        
        # Resize the face to match model input size
        resized_face = cv2.resize(face_roi, (160, 160))
        
        # Normalize pixel values (typical range 0-1)
        normalized_face = resized_face.astype("float") / 255.0
        
        # Add a batch dimension
        preprocessed_face = np.expand_dims(normalized_face, axis=0)
        
        # Make prediction using the anti-spoofing model
        preds = model.predict(preprocessed_face)[0]
        print(f"Model Prediction: {preds}")  # Debugging print
        
        # Determine label based on prediction
        if preds > 0.4:
            label = 'spoof'
            color = (0, 0, 255)  # Red for spoof
        else:
            label = 'real'
            color = (0, 255, 0)  # Green for real
        
        # Draw rectangle and label on the ORIGINAL frame (preserving its color)
        cv2.putText(original_image, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(original_image, (x, y), (x+w, y+h), color, 2)
        
        predictions.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h), 
            "label": label, 
            "confidence": float(preds)
        })

    return original_image, predictions

@app.websocket("/predict")
async def predict(websocket: WebSocket):
    """Handles WebSocket connection, processes video stream, and sends predictions."""
    await websocket.accept()
    cap = cv2.VideoCapture(0)  # Open default webcam

    # Variables to track detection results
    spoof_count = 0
    real_count = 0
    start_time = time.time()
    duration = 4  # Number of seconds to analyze
    detection_times = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame from video stream")
                break

            # Detect faces and predict liveness
            detection_start = time.time()
            annotated_frame, predictions = detect_and_predict(frame)

            # Convert frame to bytes for WebSocket transmission
            # Use BGR to JPEG conversion which preserves original colors
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            # Update counts based on predictions
            for prediction in predictions:
                if prediction["label"] == "real":
                    real_count += 1
                    spoof_count = max(0, spoof_count - 1)
                else:
                    spoof_count += 1
                    real_count = max(0, real_count - 1)

            # Record detection time
            detection_end = time.time()
            detection_time = detection_end - detection_start
            detection_times.append(detection_time)

            # Send frame and prediction data to the client
            await websocket.send_bytes(frame_bytes)
            await websocket.send_text(json.dumps(predictions, default=convert_numpy_to_json_serializable))

            # Check elapsed time and make final decision
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration:
                # Calculate average detection time
                avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
                
                # Determine final result
                final_result = "Access Granted" if real_count > spoof_count else "Try Again"
                
                # Prepare result dictionary
                result = {
                    "final_result": final_result,
                    "avg_detection_time": avg_detection_time,
                    "real_count": real_count,
                    "spoof_count": spoof_count
                }
                
                # Send final result
                await websocket.send_text(json.dumps(result, default=convert_numpy_to_json_serializable))
                break

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




















    
"""from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import img_to_array  # This line is used for future potential usage
import os
from collections import deque

# Load Face Detection Model (replace with your path)
face_cascade = cv2.CascadeClassifier("C:/face-liveliness/Anti_spoofing-model/asfr_team404-/models/haarcascade_frontalface_default.xml")

# Load Anti-Spoofing Model (adjust paths as needed)

model = load_model('C:/face-liveliness/Anti_spoofing-model/asfr_team404-/antispoofing_full_model.h5')

app = FastAPI()


def preprocess_image(frame):
 
    # Convert to RGB format (if needed)
    frame_rgb = frame[:, :, ::-1].copy()
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    return frame_rgb, gray


def detect_and_predict(frame):
    # Preprocess the image
    original_image, gray_image = preprocess_image(frame)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    # Process each detected face
    predictions = []
    for (x, y, w, h) in faces:
        face_roi = original_image[y:y + h, x:x + w]
        # Resize the face to match model input size
        resized_face = cv2.resize(face_roi, (160, 160))
        # Normalize pixel values (typical range 0-1)
        normalized_face = resized_face.astype("float") / 255.0
        # Add a batch dimension
        preprocessed_face = np.expand_dims(normalized_face, axis=0)
        # Make prediction using the anti-spoofing model
        prediction = model.predict(preprocessed_face)[0]
        label = "real" if prediction[0] < 0.4 else "spoof"
        predictions.append({"x": x, "y": y, "w": w, "h": h, "label": label})

        # Draw bounding box and label
        color = (0, 255, 0) if label == "real" else (0, 0, 255)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(original_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return original_image, predictions


@app.websocket("/predict")
async def predict(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)  # Open default webcam

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame from video stream")
                break

            # Detect faces and predict liveness
            annotated_frame, predictions = detect_and_predict(frame)

            # Convert annotated frame to bytes for WebSocket transmission
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            # Send annotated frame and prediction data to the client in separate messages
            await websocket.send_bytes(frame_bytes)
            await websocket.send_text(str(predictions))

        except WebSocketDisconnect:
            break

    cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)"""







"""from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import img_to_array  # This line is used for future potential usage
import os
from collections import deque

# Load Face Detection Model (replace with your path)
face_cascade = cv2.CascadeClassifier("C:/face-liveliness/Anti_spoofing-model/asfr_team404-/models/haarcascade_frontalface_default.xml")

# Load Anti-Spoofing Model (adjust paths as needed)
json_file = open('C:/face-liveliness/Anti_spoofing-model/asfr_team404-/finalyearproject_antispoofing_model_mobilenet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = load_model('C:/face-liveliness/Anti_spoofing-model/asfr_team404-/antispoofing_full_model.h5')

app = FastAPI()

def preprocess_image(frame):

    # Convert to RGB format (if needed)
    frame_rgb = frame[:, :, ::-1].copy()
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    return frame_rgb, gray

def detect_and_predict(frame):
 
    # Preprocess the image
    original_image, gray_image = preprocess_image(frame)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    # Process each detected face
    predictions = []
    for (x, y, w, h) in faces:
        face_roi = original_image[y:y + h, x:x + w]
        # Resize the face to match model input size
        resized_face = cv2.resize(face_roi, (160, 160))
        # Normalize pixel values (typical range 0-1)
        normalized_face = resized_face.astype("float") / 255.0
        # Add a batch dimension
        preprocessed_face = np.expand_dims(normalized_face, axis=0)
        # Make prediction using the anti-spoofing model
        prediction = model.predict(preprocessed_face)[0]
        label = "real" if prediction[0] < 0.4 else "spoof"
        predictions.append({"x": x, "y": y, "w": w, "h": h, "label": label})

    return predictions

@app.websocket("/predict")
async def predict(websocket: WebSocket):
  
    await websocket.accept()
    cap = cv2.VideoCapture(0)  # Open default webcam

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame from video stream")
                break

            # Detect faces and predict liveness
            predictions = detect_and_predict(frame)

            # Convert frame to bytes for WebSocket transmission
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Send frame and prediction data to the client in separate messages
            await websocket.send_bytes(frame_bytes)
            await websocket.send_text(str(predictions))

        except WebSocketDisconnect:
            break

    cap.release()

if __name__ == "main":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array  # This line is used for future potential usage
import os

# Load Face Detection Model (replace with your path)
face_cascade = cv2.CascadeClassifier("C:/face-liveliness/Anti_spoofing-model/asfr_team404-/models/haarcascade_frontalface_default.xml")

# Load Anti-Spoofing Model (adjust paths as needed)
model = load_model('C:/face-liveliness/Anti_spoofing-model/asfr_team404-/antispoofing_full_model.h5')

app = FastAPI()

def preprocess_image(frame):
    
    # Convert to RGB format (if needed)
    frame_rgb = frame[:, :, ::-1].copy()
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    return frame_rgb, gray

def detect_and_predict(frame):
    
    # Preprocess the image
    original_image, gray_image = preprocess_image(frame)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    # Process each detected face
    predictions = []
    for (x, y, w, h) in faces:
        face_roi = original_image[y:y + h, x:x + w]
        # Resize the face to match model input size
        resized_face = cv2.resize(face_roi, (160, 160))
        # Normalize pixel values (typical range 0-1)
        normalized_face = resized_face.astype("float") / 255.0
        # Add a batch dimension
        preprocessed_face = np.expand_dims(normalized_face, axis=0)
        # Make prediction using the anti-spoofing model
        prediction = model.predict(preprocessed_face)[0]
        label = "real" if prediction[0] < 0.5 else "spoof"
        predictions.append({"x": x, "y": y, "w": w, "h": h, "label": label})

    return predictions

@app.websocket("/predict")
async def predict(websocket: WebSocket):
 
    await websocket.accept()
    cap = cv2.VideoCapture(0)  # Open default webcam

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame from video stream")
                break

            # Detect faces and predict liveness
            predictions = detect_and_predict(frame)

            # Convert frame to bytes for WebSocket transmission
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Send frame and prediction data to the client in separate messages
            await websocket.send_bytes(frame_bytes)
            await websocket.send_text(str(predictions))

        except WebSocketDisconnect:
            break

    cap.release()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""


































from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import json

# Load Face Detection Model (replace with your path)
face_cascade = cv2.CascadeClassifier("C:/face-liveliness/Anti_spoofing-model/asfr_team404-/models/haarcascade_frontalface_default.xml")

# Load Anti-Spoofing Model (adjust paths as needed)
model = load_model('C:/face-liveliness/Anti_spoofing-model/asfr_team404-/antispoofing_full_model.h5')

app = FastAPI()

def convert_numpy_to_json_serializable(obj):
    """Convert NumPy types to standard Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def preprocess_image(frame):
    """Preprocesses the image for face detection while preserving original colors."""
    # Critical: Use the original color space of the frame
    # Do NOT convert or modify color channels
    
    # Create grayscale for face detection WITHOUT modifying original frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray

def detect_and_predict(frame):
    """Detects faces and predicts liveness using the model."""
    # Preprocess the image
    original_image, gray_image = preprocess_image(frame)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    # Process each detected face
    predictions = []
    for (x, y, w, h) in faces:
        # Extract face ROI from ORIGINAL color image
        face_roi = original_image[y:y + h, x:x + w]
        
        # Resize the face to match model input size
        resized_face = cv2.resize(face_roi, (160, 160))
        
        # Normalize pixel values (typical range 0-1)
        normalized_face = resized_face.astype("float") / 255.0
        
        # Add a batch dimension
        preprocessed_face = np.expand_dims(normalized_face, axis=0)
        
        # Make prediction using the anti-spoofing model
        preds = model.predict(preprocessed_face)[0]
        print(f"Model Prediction: {preds}")  # Debugging print
        
        # Determine label based on prediction
        if preds > 0.4:
            label = 'spoof'
            color = (0, 0, 255)  # Red for spoof
        else:
            label = 'real'
            color = (0, 255, 0)  # Green for real
        
        # Draw rectangle and label on the ORIGINAL frame (preserving its color)
        cv2.putText(original_image, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(original_image, (x, y), (x+w, y+h), color, 2)
        
        predictions.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h), 
            "label": label, 
            "confidence": float(preds)
        })

    return original_image, predictions

@app.websocket("/predict")
async def predict(websocket: WebSocket):
    """Handles WebSocket connection, processes video stream, and sends predictions."""
    await websocket.accept()
    cap = cv2.VideoCapture(0)  # Open default webcam

    # Variables to track detection results
    spoof_count = 0
    real_count = 0
    start_time = time.time()
    duration = 4  # Number of seconds to analyze
    detection_times = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame from video stream")
                break

            # Detect faces and predict liveness
            detection_start = time.time()
            annotated_frame, predictions = detect_and_predict(frame)

            # Convert frame to bytes for WebSocket transmission
            # Use BGR to JPEG conversion which preserves original colors
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            # Update counts based on predictions
            for prediction in predictions:
                if prediction["label"] == "real":
                    real_count += 1
                    spoof_count = max(0, spoof_count - 1)
                else:
                    spoof_count += 1
                    real_count = max(0, real_count - 1)

            # Record detection time
            detection_end = time.time()
            detection_time = detection_end - detection_start
            detection_times.append(detection_time)

            # Send frame and prediction data to the client
            await websocket.send_bytes(frame_bytes)
            await websocket.send_text(json.dumps(predictions, default=convert_numpy_to_json_serializable))

            # Check elapsed time and make final decision
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration:
                # Calculate average detection time
                avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
                
                # Determine final result
                final_result = "Access Granted" if real_count > spoof_count else "Try Again"
                
                # Prepare result dictionary
                result = {
                    "final_result": final_result,
                    "avg_detection_time": avg_detection_time,
                    "real_count": real_count,
                    "spoof_count": spoof_count
                }
                
                # Send final result
                await websocket.send_text(json.dumps(result, default=convert_numpy_to_json_serializable))
                break

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)