# CARLA Autonomous Driving with Object Detection & Federated Learning

This README explains how to set up and run a CARLA simulation with autonomous vehicles, integrated YOLOv8 object detection, and federated learning for updating a global model.  It provides step by step instructions for starting the CARLA simulator, running Python scripts, and saving detection logs.  By following this guide, anyone  can replicate the full autonomous driving and federated learning framework on their server.

This repo contains:

- Autonomous driving in CARLA using `CAD_FL.py` as base.  
- YOLOv8 object detection:
  - `yolov8m.pt` → COCO pretrained (8 classes)  
  - `global_model.pt` → custom trained global model (`pole` class)  
- Federated Learning aggregation (`3.fedFL.py`) to update the global model.  

## 📂 Repository Structure
carla-habeslab/
├── README.md
├── .gitignore
├── docker/
│ └── run_carla.sh # Start CARLA simulator
└── python/
├── CAD_FL.py # Main script (object detection + CARLA control)
├── 3.fedFL.py # Federated Learning aggregation
├── run_client.sh # Runs both CAD_FL.py + 3.fedFL.py
├── requirements.txt # Python dependencies
└── logs/ # Images and YOLO labels