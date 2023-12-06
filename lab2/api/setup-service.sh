#!/bin/bash

# Require root privileges to run this script
if [ "$(id -u)" != "0" ]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi

# Define the service file content
SERVICE_FILE_CONTENT="[Unit]
Description=ROI Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/scalable/scalable-ml/lab2/api
ExecStartPre=/usr/bin/python -m pip install -r requirements.txt
ExecStart=/usr/bin/python /root/scalable/scalable-ml/lab2/api/app.py
Restart=on-failure

[Install]
WantedBy=multi-user.target"

# Write the service file content to roi.service
sudo echo "$SERVICE_FILE_CONTENT" > /etc/systemd/system/roi.service

# Reload the systemd daemon to recognize the new service
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable roi.service
sudo systemctl start roi.service

echo "ROI service setup complete."