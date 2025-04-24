# onnx_integration.py

import cv2
import numpy as np
import os
import time
import datetime
import logging
import onnxruntime as ort
from pathlib import Path

class ONNXWorkerDetector:
    def __init__(self, model_path, confidence_threshold=0.5, img_size=640):
        """
        Initialize the ONNX-based worker detector.
        
        Args:
            model_path: Path to the ONNX model
            confidence_threshold: Minimum confidence score for detections
            img_size: Input image size for the model
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.img_size = img_size
        
        # Initialize ONNX Runtime session
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            print(f"ONNX model loaded successfully from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def preprocess(self, img):
        """
        Preprocess an image for the ONNX model.
        
        Args:
            img: Input image
            
        Returns:
            processed_img: Preprocessed image
        """
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        
        # HWC to CHW (height, width, channels to channels, height, width)
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def detect(self, img):
        """
        Detect workers in an image.
        
        Args:
            img: Input image
            
        Returns:
            detections: List of detection results (x1, y1, x2, y2, confidence, class_id)
        """
        # Get original dimensions
        height, width = img.shape[:2]
        
        # Preprocess image
        input_tensor = self.preprocess(img)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Process outputs
        detections = []
        
        # YOLOv5 ONNX output format: [batch_id, x, y, w, h, confidence, class_id, ...]
        for detection in outputs[0][0]:
            confidence = float(detection[4])
            class_id = int(detection[5])
            
            # Filter by confidence and class (0 = person)
            if confidence > self.confidence_threshold and class_id == 0:
                # Convert from center_x, center_y, width, height to x1, y1, x2, y2
                center_x, center_y, w, h = detection[0:4]
                
                # Convert from normalized coordinates [0,1] to pixel coordinates
                x1 = int((center_x - w/2) * width)
                y1 = int((center_y - h/2) * height)
                x2 = int((center_x + w/2) * width)
                y2 = int((center_y + h/2) * height)
                
                detections.append((x1, y1, x2, y2, confidence, class_id))
        
        return detections