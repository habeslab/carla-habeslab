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
│   └── yolov8m.pt           # Default YOLOv8 model for 80 COCO classes
│   └── coco.names           # COCO class names (used with yolov8m.pt)
│   └── custom.names         # Custom class name (used with local model)

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
   - Runs YOLOv8 detection for both COCO classes (`yolov8m.pt`) and custom pole class (`best.pt`).  
   - Saves camera frames and detection labels.  
3. `3.fedFL.py` runs:
   - Aggregates local models from vehicles to update the global model.  
   - Runs in the background continuously.

**Saved logs**:
```text
python/logs/vehicle_<id>/
├── images/ # Camera frames captured by the vehicle
└── labels/ # YOLO labels (custom only) for each frame
```

- Frames and labels are saved every 10 frames by default (configurable in `CAD_FL.py`).  

## Step 4: YOLOv8 Models

This framework uses two YOLOv8 models for object detection in CARLA:

| Model | Purpose | Location |
|-------|---------|---------|
| `yolov8m.pt` | Pretrained on COCO dataset (80 classes) for general object detection | `python/` |
| `best.pt` | Custom-trained local model for detecting `pole` | `python/pole/weights/best.pt` |

This project uses explicit class name files during detection:

- `coco.names`  
  - Contains the 80 COCO class labels  
  - Used by `yolov8m.pt` for general object detection  

- `custom.names`  
  - Contains custom class labels (e.g., `pole`)  
  - Used by the federated/global YOLO model (`global_model.pt`)  

These files are loaded in `CAD_FL.py` to correctly map predicted class IDs to human readable labels during real-time detection and logging.


## Step 5: Notes & Configuration

- **Camera & Frame Settings**:  
  - Resolution: `640x480`  
  - Field of View (FOV): `90`  
  - Frames saved every 10 frames (configurable via `ObjectDetection.SAVE_INTERVAL` in `CAD_FL.py`)  

- **Federated Learning Configuration** (`3.fedFL.py`):
  - `vehicles` → List of vehicle IDs used in federated learning
  - `base_model_path` → Path to local YOLO model (`best.pt`)
  - `global_model_path` → Path to save aggregated global model
  - `local_epochs` → Number of epochs for local training per vehicle
  - `max_images_per_round` → Minimum images per vehicle before aggregation
  - `mu`, `head_alpha`, `clip_max` → FedProx aggregation parameters

- **Custom Classes**:
  - COCO classes for general detection (`yolov8m.pt`)  
  - Pole class for local/global detection (`best.pt` / `global_model.pt`)  

- **Logs and Outputs**:
  - All images and labels are saved in `python/logs/vehicle_<id>/images` and `/labels`  
  - Global model is updated in `python/global/global_model.pt` after each aggregation round  

- **Best Practices**:
  - Start CARLA simulator first (Step 1)  
  - Ensure `run_client.sh` is executed in a Python environment with all dependencies installed  
  - Avoid running multiple `CAD_FL.py` instances simultaneously unless needed  
  - Check disk space as images accumulate over time  

## Step 6: Troubleshooting

- **CARLA Not Starting**: Make sure Docker is running and ports are free  
- **YOLO Model Errors**: Ensure `yolov8m.pt` and `best.pt` exist in their correct paths  
- **No Images Saved**: Check `ObjectDetection.SAVE_INTERVAL` and camera attachment to vehicles  
- **Federated Learning Not Updating**: Verify `max_images_per_round` is met for all vehicles  

## References & Credits

- [CARLA Simulator](https://carla.org/) – Open-source autonomous driving simulator  
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics) – Object detection framework   
- This repository integrates YOLOv8 detection into CARLA’s `automatic_control.py` example, enabling real-time detection with federated learning updates.
