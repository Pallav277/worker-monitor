# worker_monitor.py

import cv2
import numpy as np
import os
import time
import datetime
import torch
import logging
from pathlib import Path
from onnx_integration import ONNXWorkerDetector

# Configure logging
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / f"worker_activity_{datetime.datetime.now().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

class WorkerMonitor:
    def __init__(self, camera_id=0, model_path='./models/yolo11s.pt', 
                 confidence_threshold=0.5, activity_threshold=50,
                 idle_timeout=5, roi_coordinates=None):
        """
        Initialize the worker monitoring system.
        
        Args:
            camera_id: ID of the camera device (default: 0)
            model_path: Path to the YOLO11s model file
            confidence_threshold: Minimum confidence score for detections
            activity_threshold: Pixel difference threshold to detect movement
            idle_timeout: Seconds of inactivity before considered idle
            roi_coordinates: Region of interest coordinates (x1, y1, x2, y2) or None for full frame
        """
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.activity_threshold = activity_threshold
        self.idle_timeout = idle_timeout
        self.roi_coordinates = roi_coordinates
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Get frame dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set ROI to full frame if not specified
        if self.roi_coordinates is None:
            self.roi_coordinates = (0, 0, self.frame_width, self.frame_height)
        
        # Load YOLO11s model for person detection
        self.model_type = None
        try:
            if model_path.lower().endswith('.pt'):
                self.model = torch.hub.load('ultralytics/yolo11s', 'custom', path=model_path)
                self.model.conf = confidence_threshold
                self.model.classes = [0]  # Only detect people (class 0 in COCO)
                self.model_type = 'pt'
                logging.info("Loaded YOLO11s model from PyTorch")
            elif model_path.lower().endswith('.onnx'):
                self.model = ONNXWorkerDetector(model_path=model_path, confidence_threshold=confidence_threshold)
                self.model_type = 'onnx'
                logging.info("Loaded YOLO11s model from ONNX")
            else:
                raise ValueError("Unsupported model format. Use .pt or .onnx")
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            print(f"Failed to load model: {e}")
            print("Falling back to basic motion detection without person detection")
            self.model = None
            self.model_type = None
        
        # State tracking
        self.last_activity_time = time.time()
        self.previous_frame = None
        self.current_state = "INITIALIZING"
        self.frame_counter = 0
        
        # Activity tracking
        self.last_state_change = time.time()
        self.consecutive_active_frames = 0
        self.consecutive_idle_frames = 0
        
        logging.info("Worker monitoring system initialized")
    
    def process_frame(self, frame):
        """
        Process a single frame from the video stream.
        
        Args:
            frame: Input video frame
            
        Returns:
            processed_frame: Frame with annotations
            is_active: Boolean indicating if worker is active
        """
        # Extract ROI
        x1, y1, x2, y2 = self.roi_coordinates
        roi = frame[y1:y2, x1:x2]
        
        # Create a copy for visualization
        display_frame = frame.copy()
        
        # Draw ROI rectangle
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize previous frame if not available
        if self.previous_frame is None:
            self.previous_frame = gray
            return display_frame, False
        
        # Calculate absolute difference between current and previous frame
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate the thresholded image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours on thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Measure motion intensity (sum of pixel differences)
        motion_intensity = np.sum(frame_delta) / 255
        
        # Person detection using YOLO11s
        person_detected = False
        if self.model is not None:
            if self.model_type == 'pt':
                # Pytorch model detection
                results = self.model(roi)
                
                # Extract detection results
                if len(results.pred[0]) > 0:
                    person_detected = True
                    
                    # Draw bounding boxes
                    for *box, conf, cls in results.pred[0]:
                        x_min, y_min, x_max, y_max = [int(val) for val in box]
                        # Adjust coordinates to original frame
                        cv2.rectangle(display_frame, 
                                    (x_min + x1, y_min + y1), 
                                    (x_max + x1, y_max + y1), 
                                    (0, 0, 255), 2)
                        cv2.putText(display_frame, f"Person: {conf:.2f}", 
                                (x_min + x1, y_min + y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
        elif self.model_type == 'onnx':
            # ONNX model detection
            detections = self.model.detect(roi)
            
            # Process detection results
            if len(detections) > 0:
                person_detected = True
                
                # Draw bounding boxes
                for x_min, y_min, x_max, y_max, conf, cls in detections:
                    # Adjust coordinates to original frame
                    cv2.rectangle(display_frame, 
                                (x_min + x1, y_min + y1), 
                                (x_max + x1, y_max + y1), 
                                (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Person: {conf:.2f}", 
                            (x_min + x1, y_min + y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        else:
            # Fallback to motion-based detection if model is not available
            person_detected = motion_intensity > self.activity_threshold
        
        # Determine activity based on motion and person detection
        is_active = person_detected and motion_intensity > self.activity_threshold
        
        # Update consecutive frames counters
        if is_active:
            self.consecutive_active_frames += 1
            self.consecutive_idle_frames = 0
            self.last_activity_time = time.time()
        else:
            self.consecutive_idle_frames += 1
            self.consecutive_active_frames = 0
        
        # Display activity status
        status_text = "ACTIVE" if is_active else "IDLE"
        cv2.putText(display_frame, f"Status: {status_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Motion: {motion_intensity:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Update previous frame
        self.previous_frame = gray
        
        return display_frame, is_active
    
    def update_state(self, is_active):
        """
        Update the overall state based on activity detection.
        
        Args:
            is_active: Boolean indicating if worker is currently active
        """
        current_time = time.time()
        
        # State transitions
        if self.current_state == "INITIALIZING":
            if self.frame_counter > 30:  # Wait for 30 frames before determining state
                self.current_state = "WORKING" if is_active else "IDLE"
                logging.info(f"Initial state: {self.current_state}")
                self.last_state_change = current_time
                
        elif self.current_state == "WORKING":
            # If idle for more than idle_timeout seconds, transition to IDLE
            if not is_active and (current_time - self.last_activity_time) > self.idle_timeout:
                self.current_state = "IDLE"
                logging.info(f"State change: WORKING -> IDLE")
                self.last_state_change = current_time
                
        elif self.current_state == "IDLE":
            # If active for at least 2 seconds (assuming 15fps), transition to WORKING
            if is_active and self.consecutive_active_frames > 30:
                self.current_state = "WORKING"
                logging.info(f"State change: IDLE -> WORKING")
                self.last_state_change = current_time
        
        self.frame_counter += 1
    
    def run(self):
        """Main execution loop for the monitoring system."""
        print("Starting worker activity monitoring...")
        logging.info("Monitoring started")
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame. Retrying...")
                    time.sleep(1)
                    continue
                
                # Process the frame
                processed_frame, is_active = self.process_frame(frame)
                
                # Update state
                self.update_state(is_active)
                
                # Display state and time info
                cv2.putText(processed_frame, f"State: {self.current_state}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(processed_frame, current_time, (10, processed_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Display the frame
                cv2.imshow("Worker Activity Monitor", processed_frame)
                
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Monitoring stopped by user")
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            logging.error(f"Error: {e}")
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()
            logging.info("Monitoring stopped")
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()


if __name__ == "__main__":
    try:
        # Create and run the worker monitor
        # You can adjust these parameters as needed
        monitor = WorkerMonitor(
            camera_id=0,  # Use 0 for default camera
            model_path='./models/yolo11s.pt',  # Path to YOLO11s model (Accepts .pt or .onnx)
            confidence_threshold=0.45,
            activity_threshold=3000,
            idle_timeout=5,
            roi_coordinates=None  # Set to None to use full frame
        )
        monitor.run()
    except Exception as e:
        logging.error(f"Application error: {e}")
        print(f"Error: {e}")