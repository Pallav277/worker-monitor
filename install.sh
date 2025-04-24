#!/bin/bash

# Installation and Setup script for Worker Activity Monitoring System

# Exit on error
set -e

echo "Installing Worker Activity Monitoring System..."

# Create directory structure
mkdir -p ../worker-monitor/models
mkdir -p ../worker-monitor/logs

# Move to the directory
cd ../worker-monitor

# Install dependencies
echo "Installing dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies. Please check your internet connection and try again."
    exit 1
fi

# Download YOLO11s model
echo "Setting up model..."
python3 setup_model.py

# Setup systemd service
echo "Setting up systemd service..."
sudo cp worker-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable worker-monitor.service

echo "Installation complete!"
echo "To start the service manually, run: sudo systemctl start worker-monitor.service"
echo "To check status, run: sudo systemctl status worker-monitor.service"
echo "To view logs, run: sudo journalctl -u worker-monitor.service"