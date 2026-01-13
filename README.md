# CARLA Autonomous Driving with Object Detection & Federated Learning

This README explains how to set up and run a CARLA simulation with autonomous vehicles, integrated YOLOv8 object detection, and federated learning for updating a global model.  It provides step by step instructions for starting the CARLA simulator, running Python scripts, and saving detection logs.  By following this guide, anyone  can replicate the full autonomous driving and federated learning framework on their server.

This repo contains:

- Autonomous driving in CARLA using `CAD_FL.py` as base.  
- YOLOv8 object detection:
  - `yolov8m.pt` → COCO pretrained (80 classes)  
  - `global_model.pt` → custom trained global model (`pole` class)  
- Federated Learning aggregation (`3.fedFL.py`) to update the global model.  

## Repository Structure
carla-habeslab/
├── README.md
├── .gitignore
├── docker/
│   └── run_carla.sh        # Script to start CARLA simulator
└── python/
    ├── CAD_FL.py           # Main script (object detection + CARLA control)
    ├── 3.fedFL.py          # Federated Learning aggregation script
    ├── run_client.sh       # Runs both CAD_FL.py + 3.fedFL.py
    └── pole/               # Folder containing trained local models
        └── best.pt         #local model file
├── requirements.txt        # Python dependencies

## Step 1: Start CARLA Simulator

Open Terminal 1 and run:

```bash
cd carla-habeslab/docker
./run_carla.sh

## Step 2: Run Object Detection + Federated Learning

Open Terminal 2 and run:
```bash
cd carla-habeslab/python
./run_client.sh


## YOLOv8 Models

This framework uses two YOLOv8 models for object detection in CARLA:

| Model | Purpose | Location |
|-------|---------|---------|
| `yolov8m.pt` | Pretrained on COCO dataset (80 classes) for general object detection | `python/` |
| `best.pt` | Custom-trained local model for detecting `pole` | `python/pole/weights/best.pt` |

Notes:

- `CAD_FL.py` loads both models per vehicle:
  - `yolov8m.pt` → general object detection
  - `best.pt` → custom detection for your pole class
- Detection results are:
  - Visualized in CARLA  
  - Saved in logs (`python/logs/vehicle_<id>/images` and `/labels`)  
- The federated learning aggregation updates the global model based on these local detections (`3.fedFL.py`).
