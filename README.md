# CARLA Autonomous Driving with Object Detection & Federated Learning

This README explains how to set up and run a CARLA simulation with autonomous vehicles, integrated YOLOv8 object detection, and federated learning for updating a global model.  It provides step by step instructions for starting the CARLA simulator, running Python scripts, and saving detection logs.  By following this guide, anyone  can replicate the full autonomous driving and federated learning framework on their server.

This repo contains:

- Autonomous driving in CARLA using `CAD_FL.py` as base.  
- YOLOv8 object detection:
  - `yolov8m.pt` → COCO pretrained (80 classes)  
  - `global_model.pt` → custom trained global model (`pole` class)  
- Federated Learning aggregation (`3.fedFL.py`) to update the global model.  

## Repository Structure
```text
carla-habeslab/
├── README.md                # This file
├── .gitignore               # Git ignore rules
├── docker/
│   └── run_carla.sh         # Script to start CARLA simulator in Docker
├── python/
│   ├── CAD_FL.py            # Main script (object detection + CARLA control)
│   ├── 3.fedFL.py           # Federated Learning aggregation script
│   ├── run_client.sh        # Runs both CAD_FL.py + 3.fedFL.py
│   ├── requirements.txt     # Python dependencies
│   └── pole/                # Folder containing trained local models
│       └── weights/
│           └── best.pt      # Locally trained YOLO model for `pole` class
│   └── yolov8m.pt           # Default YOLOv8 model for 8 COCO classes

```
               
## Step 1: Start CARLA Simulator

Open Terminal 1 and run:

```bash
cd carla-habeslab/docker
./run_carla.sh
```
## Step 2: Run Object Detection + Federated Learning

Open Terminal 2 and run:
```bash
cd carla-habeslab/python
./run_client.sh
```

## Step 3: Logs and Output

When you run `run_client.sh`:

1. The Python virtual environment is created and dependencies are installed.  
2. `CAD_FL.py` runs:
   - Starts autonomous driving in CARLA.  
   - Runs YOLOv8 detection for both COCO classes** (`yolov8m.pt`) and **custom pole class** (`best.pt`).  
   - Saves camera frames and detection labels.  
3. `3.fedFL.py` runs:
   - Aggregates local models from vehicles to update the global model.  
   - Runs in the background continuously.

**Saved logs**:
```text
python/logs/vehicle_<id>/
├── images/ # Camera frames captured by the vehicle
└── labels/ # YOLO labels (COCO + custom) for each frame
```

- Frames and labels are saved every **10 frames** by default (configurable in `CAD_FL.py`).  

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



## Step 3: Run Object Detection + Federated Learning