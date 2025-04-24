# Worker Activity Monitoring System

This project implements a standalone computer vision system that detects whether a worker is actively working. The system uses a camera to capture video, processes the frames to detect worker presence and activity, and logs the activity states.

## Features

- Real-time worker detection using YOLO11s
- Motion-based activity analysis
- Automatic startup on boot (systemd service)
- Activity state logging
- Low hardware requirements
- Configurable parameters

## Hardware Requirements

- Any computer with a camera (USB webcam or built-in camera)
- Recommended: Raspberry Pi 4 (2GB RAM or more)
- Camera with a clear view of the work area

## Installation

### Prerequisites
- Python 3.x
- Git
- USB webcam or built-in camera

### Installation Steps

1. Clone the repository:
   ```
   [git clone https://github.com/yourusername/worker-monitor](https://github.com/Pallav277/worker-monitor.git)
   cd worker-monitor
   ```

2. Environment Setup:
   - For Local Testing (PC/Laptop):
     ```
     python3 -m venv venv
     source venv/bin/activate  # On Linux/Mac
     # Note: Use 'deactivate' when you want to exit the virtual environment
     ```
   - For Single-Board Computers (like Raspberry Pi):
     - No virtual environment is needed, proceed to the next step

3. Modify the Service File:
   - Open `worker-monitor.service`
   - Update the following paths:
     - `WorkingDirectory`: Set to your project directory path
     - `ExecStart`: Set to your Python script path
   Example:
   ```
   WorkingDirectory=/your/path/to/worker-monitor
   ExecStart=/usr/bin/python3 /your/path/to/worker-monitor/worker_monitor.py
   ```

4. Run the Installation Script:
   ```
   chmod +x install.sh
   ./install.sh
   ```
   This will:
   - Install all required dependencies
   - Set up the project directories
   - Download the necessary model files
   - Configure and start the systemd service

## Configuration

Edit the `worker_monitor.py` file to adjust the following parameters:

- `camera_id`: ID of the camera device (default: 0)
- `model_path`: Path to the YOLO11s model file
- `confidence_threshold`: Minimum confidence score for detections
- `activity_threshold`: Pixel difference threshold to detect movement
- `idle_timeout`: Seconds of inactivity before considered idle
- `roi_coordinates`: Region of interest coordinates (x1, y1, x2, y2)

## Usage

The system will start automatically on boot once installed. To manually control the service:

- Start: `sudo systemctl start worker-monitor.service`
- Stop: `sudo systemctl stop worker-monitor.service`
- Check status: `sudo systemctl status worker-monitor.service`
- View logs: `sudo journalctl -u worker-monitor.service`

## Troubleshooting

1. **Camera not found**: Make sure your camera is properly connected and check the `camera_id` parameter.
2. **Model not loading**: Verify that the model file exists in the correct path. Run `setup_model.py` to download it.
3. **Service not starting**: Check the system logs with `sudo journalctl -u worker-monitor.service`.
4. **High CPU usage**: Try using a smaller YOLO11s model variant if available.

## License

MIT License
