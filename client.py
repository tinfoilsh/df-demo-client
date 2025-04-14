#!/usr/bin/env python3
"""
Real-time Deepfake Detector Client
----------------------------------
This application captures your screen during video calls and sends
frames to a deepfake detection inference server for analysis.
"""

import os
import sys
import time
import argparse
import threading
import numpy as np
import cv2
import requests
import base64
import json
from PIL import ImageGrab, Image
from datetime import datetime

# Global variables
detection_active = False
capture_thread = None
detection_results = []
current_frame = None
save_dir = "detected_frames"
server_url = "http://localhost:8001"  # Default server URL

# Try to load server URL from config file
CONFIG_FILE = "client_config.json"
try:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            if 'server_url' in config:
                server_url = config['server_url']
                print(f"Loaded server URL from config: {server_url}")
except Exception as e:
    print(f"Error loading config: {e}")

def encode_image(image):
    """
    Encode an image as base64 for API transmission
    
    Args:
        image: The image as numpy array
        
    Returns:
        str: Base64 encoded image string
    """
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        return None
    
    return base64.b64encode(encoded_image).decode('utf-8')

def analyze_frame(frame):
    """
    Send a frame to the inference server for deepfake analysis
    
    Args:
        frame: The image frame to analyze (numpy array)
        
    Returns:
        dict: Analysis results with scores for each category
    """
    try:
        # Encode the image
        encoded_image = encode_image(frame)
        if encoded_image is None:
            return {"error": "Failed to encode image"}
        
        # Prepare the request
        url = f"{server_url}/analyze"
        payload = json.dumps({"image": encoded_image})
        headers = {'Content-Type': 'application/json'}
        
        # Send the request to the server
        response = requests.post(url, headers=headers, data=payload, timeout=30)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()["results"]
        else:
            error_msg = response.json().get("error", f"Server returned status code {response.status_code}")
            return {"error": error_msg}
    
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def check_server_health():
    """
    Check if the inference server is running and healthy
    
    Returns:
        bool: True if server is healthy, False otherwise
    """
    try:
        print(f"Checking server health at {server_url}/health...")
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("status") == "healthy" and health_data.get("model_loaded"):
                print("Server is healthy and model is loaded")
                return True
            else:
                print(f"Server is not ready: {health_data}")
                return False
        else:
            print(f"Server health check failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        return False

def detect_face(frame):
    """
    Detect faces in the frame and return the largest face region
    
    Args:
        frame: The image frame (numpy array)
        
    Returns:
        tuple: (x, y, w, h) coordinates of the largest face, or None if no face detected
    """
    try:
        # Load the pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # If faces are detected, return the largest one
        if len(faces) > 0:
            # Find the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            return largest_face
        
        return None
    
    except Exception as e:
        print(f"Face detection error: {e}")
        return None

def crop_to_face(frame, add_margin=True):
    """
    Crop the frame to focus on the detected face
    
    Args:
        frame: The image frame (numpy array)
        add_margin: Whether to add margin around the face
        
    Returns:
        numpy array: Cropped image around the face, or original frame if no face detected
    """
    face_coords = detect_face(frame)
    
    if face_coords is not None:
        x, y, w, h = face_coords
        
        if add_margin:
            # Add 20% margin
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            
            # Calculate new coordinates with margin
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(frame.shape[1], x + w + margin_x)
            y2 = min(frame.shape[0], y + h + margin_y)
            
            # Crop the image
            return frame[y1:y2, x1:x2]
        else:
            # Crop without margin
            return frame[y:y+h, x:x+w]
    
    return frame

def capture_screen(region=None, interval=1.0, face_only=True):
    """
    Continuously capture the screen and analyze frames
    
    Args:
        region: The region to capture (left, top, right, bottom) or None for full screen
        interval: Time interval between captures in seconds
        face_only: Whether to crop to detected faces only
    """
    global detection_active, current_frame, detection_results
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting screen capture with {interval:.1f}s interval")
    print("Press Ctrl+C to stop")
    
    try:
        while detection_active:
            # Capture screen
            screenshot = ImageGrab.grab(bbox=region)
            
            # Convert to numpy array
            frame = np.array(screenshot)
            
            # Convert RGB to BGR (for OpenCV)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Store the current frame for display
            current_frame = frame.copy()
            
            # Crop to face if enabled
            if face_only:
                analysis_frame = crop_to_face(frame)
            else:
                analysis_frame = frame
            
            # Analyze the frame
            results = analyze_frame(analysis_frame)
            
            # Store results
            if "error" not in results:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                detection_results.append({
                    "timestamp": timestamp,
                    "results": results
                })
                
                # Find the highest score
                highest_category = max(results.items(), key=lambda x: x[1])
                
                # Print results
                print(f"[{timestamp}] Classification: {highest_category[0]} ({highest_category[1]*100:.2f}%)")
                
                # Check if it's a deepfake or AI-generated with high confidence
                if (highest_category[0] == "Deepfake" or highest_category[0] == "Artificial") and highest_category[1] > 0.7:
                    print(f"⚠️ ALERT: Potential {highest_category[0]} detected with {highest_category[1]*100:.2f}% confidence!")
                    
                    # Save the frame
                    filename = f"{save_dir}/detected_{highest_category[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, analysis_frame)
                    print(f"Frame saved to {filename}")
            else:
                print(f"Error: {results['error']}")
            
            # Wait for the next interval
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("Screen capture stopped")
    except Exception as e:
        print(f"Error during screen capture: {e}")
    finally:
        detection_active = False

def start_detection(region=None, interval=1.0, face_only=True):
    """Start the detection process in a separate thread"""
    global detection_active, capture_thread
    
    if detection_active:
        print("Detection is already running")
        return
    
    # Check if server is healthy
    if not check_server_health():
        print("Failed to connect to inference server. Cannot start detection.")
        return
    
    # Start detection
    detection_active = True
    capture_thread = threading.Thread(
        target=capture_screen,
        args=(region, interval, face_only)
    )
    capture_thread.daemon = True
    capture_thread.start()

def stop_detection():
    """Stop the detection process"""
    global detection_active
    
    if not detection_active:
        print("Detection is not running")
        return
    
    detection_active = False
    if capture_thread:
        capture_thread.join(timeout=2.0)
    
    print("Detection stopped")

def display_results():
    """Display the detection results"""
    if not detection_results:
        print("No detection results available")
        return
    
    print("\n===== DETECTION RESULTS =====")
    for i, result in enumerate(detection_results[-10:], 1):  # Show last 10 results
        print(f"{i}. [{result['timestamp']}]")
        
        # Find highest score
        highest = max(result['results'].items(), key=lambda x: x[1])
        
        print(f"   Classification: {highest[0]} ({highest[1]*100:.2f}%)")
        print(f"   Detailed scores:")
        
        for category, score in result['results'].items():
            print(f"     - {category}: {score*100:.2f}%")
        
        print()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Real-time Deepfake Detector Client")
    
    parser.add_argument("--region", type=str, help="Screen region to capture (left,top,right,bottom)")
    parser.add_argument("--interval", type=float, default=1.0, help="Capture interval in seconds")
    parser.add_argument("--no-face-crop", action="store_true", help="Disable face cropping")
    parser.add_argument("--server", type=str, help=f"Inference server URL (default: {server_url})")
    
    return parser.parse_args()

def main():
    """Main function"""
    global server_url
    
    args = parse_arguments()
    
    # Set server URL if provided via command line
    if args.server:
        server_url = args.server
    
    # Parse region if provided
    region = None
    if args.region:
        try:
            region = tuple(map(int, args.region.split(',')))
            if len(region) != 4:
                raise ValueError("Region must have 4 values")
        except Exception as e:
            print(f"Error parsing region: {e}")
            print("Using full screen instead")
            region = None
    
    print("Real-time Deepfake Detector Client")
    print("----------------------------------")
    print(f"Inference server: {server_url}")
    print(f"Capture region: {'Full screen' if region is None else region}")
    print(f"Capture interval: {args.interval} seconds")
    print(f"Face cropping: {'Disabled' if args.no_face_crop else 'Enabled'}")
    print()
    
    try:
        # Start detection
        start_detection(region, args.interval, not args.no_face_crop)
        
        # Keep the main thread running
        while detection_active:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        stop_detection()
        display_results()

if __name__ == "__main__":
    main()
