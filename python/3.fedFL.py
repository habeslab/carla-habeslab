#!/usr/bin/env python3
"""
Federated YOLOv8 (FedProx) for CARLA clients.
- State_dict aggregation
- Local training if 50 images per vehicle
- Empty round detection
- Automatic continuation from last round
"""

import os
import time
import shutil
import torch
import yaml
from ultralytics import YOLO

# =========================
# CONFIGURATION
# =========================
vehicles = ["vehicle_0", "vehicle_1", "vehicle_2"]
base_path = "/home/habes/Documents/PythonAPI/examples/logs"
base_model_path = "/home/habes/Documents/PythonAPI/examples/runs/detect/train/weights/best.pt"
global_model_path = "/home/habes/Documents/PythonAPI/examples/global/global_model.pt"

empty_rounds = 0
local_epochs = 20
batch = 32
imgsz = 768
mu = 0.2
head_alpha = 0.25
clip_max = 10.0
max_images_per_round = 5
sleep_if_empty = 30
sleep_between_rounds = 30
custom_names_file = "/home/habes/Documents/PythonAPI/examples/custom.names"
with open(custom_names_file, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# =========================
# HELPERS
# =========================
def get_next_round_index(base_path, vehicles):
    round_indices = []
    for vehicle in vehicles:
        vpath = os.path.join(base_path, vehicle)
        if not os.path.exists(vpath):
            continue
        for d in os.listdir(vpath):
            if d.startswith("round_"):
                try:
                    ridx = int(d.split("_")[1])
                    round_indices.append(ridx)
                except:
                    pass
    if not round_indices:
        return 0
    next_idx = max(round_indices) + 1
    print(f"ℹ️ Detected existing rounds up to {max(round_indices)} → starting from round {next_idx}")
    return next_idx

def update_dataset_yaml(vehicle, round_idx):
    path = f"{base_path}/{vehicle}/round_{round_idx}"
    os.makedirs(path, exist_ok=True)
    yaml_file = f"{path}/dataset.yaml"
    yaml.dump({'path': path, 'train': 'images', 'val': 'images', 'names': class_names},
              open(yaml_file, 'w'))
    return yaml_file

def is_finite(t):
    return torch.isfinite(t).all().item()

def safe_fedprox_update(local_tensor, global_tensor):
    v_new = (1.0 - mu) * local_tensor + mu * global_tensor
    v_new = torch.nan_to_num(v_new, nan=0.0, posinf=1e3, neginf=-1e3)
    return torch.clamp(v_new, -clip_max, clip_max)

def weighted_avg(tensors, weights):
    stacked = torch.stack([w * t for t, w in zip(tensors, weights)], dim=0)
    return stacked.sum(dim=0)

def load_base_or_global():
    if os.path.exists(global_model_path):
        print(f"ℹ️ Loaded previous global model from {global_model_path}")
        ckpt = torch.load(global_model_path, map_location='cpu')
        if 'state_dict' in ckpt:
            return ckpt['state_dict']
    print("ℹ️ Using base model as initial global state")
    m = YOLO(base_model_path)
    return m.model.state_dict()

def is_head_key(k):
    lower = k.lower()
    return any(x in lower for x in ['cls', 'reg', 'anchor', 'detect', 'head', 'm.22', 'm.23', 'm.24'])

def debug_states(global_state, local_states, local_names):
    print("=== Global keys & shapes (first 30) ===")
    for k, v in list(global_state.items())[:30]:
        print(f"  {k:60s} {tuple(v.shape)}")
    for i, s in enumerate(local_states):
        print(f"\n--- Local {local_names[i]} mismatches / shape diffs ---")
        missing = [k for k in s.keys() if k not in global_state]
        shape_mismatch = [(k, tuple(s[k].shape), tuple(global_state[k].shape)) 
                          for k in s.keys() if k in global_state and s[k].shape != global_state[k].shape]
        if missing:
            print(f"  MISSING in global ({len(missing)} keys), example: {missing[:5]})")
        else:
            print("  No keys missing in global.")
        if shape_mismatch:
            print(f"  SHAPE MISMATCHES (examples 10): {shape_mismatch[:10]}")
        else:
            print("  No shape mismatches.")

# =========================
# MAIN FEDERATED LOOP
# =========================
round_idx = get_next_round_index(base_path, vehicles)
print(f"▶️ Starting federated loop from round {round_idx}")

while True:
    print(f"\n===================\n🔄 FedProx Round {round_idx}\n===================")
    local_infos = []

    # Load global/base state (dict)
    global_state = load_base_or_global()
    if not global_state:
        print("❌ No valid global/base state found. Exiting.")
        break

    # =========================
    # MOVE NEW IMAGES TO PENDING
    # =========================
    for vehicle in vehicles:
        src_imgs = f"{base_path}/{vehicle}/images"
        src_lbls = f"{base_path}/{vehicle}/labels"
        pending_imgs = f"{base_path}/{vehicle}/pending_images"
        pending_lbls = f"{base_path}/{vehicle}/pending_labels"
        os.makedirs(pending_imgs, exist_ok=True)
        os.makedirs(pending_lbls, exist_ok=True)
        for folder, dest in [(src_imgs, pending_imgs), (src_lbls, pending_lbls)]:
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    src_path = os.path.join(folder, f)
                    dst_path = os.path.join(dest, f)
                    if not os.path.exists(dst_path):
                        shutil.move(src_path, dst_path)

    # =========================
    # WAIT UNTIL ALL VEHICLES HAVE 50 IMAGES
    # =========================
    all_ready = True
    for vehicle in vehicles:
        pending_imgs = f"{base_path}/{vehicle}/pending_images"
        total_pending = len(os.listdir(pending_imgs)) if os.path.exists(pending_imgs) else 0
        if total_pending < max_images_per_round:
            print(f"⚠️ {vehicle} waiting for more images ({total_pending}/{max_images_per_round})")
            all_ready = False

    if not all_ready:
        empty_rounds += 1
        print(f"⚠️ Not all vehicles ready this round (empty_rounds={empty_rounds})")
        if empty_rounds >= 2:
            print("ℹ️ Stopping federated training due to 2 consecutive empty rounds")
            break
        time.sleep(sleep_if_empty)
        continue
    empty_rounds = 0

    # =========================
    # LOCAL TRAINING
    # =========================
    for vehicle in vehicles:
        pending_imgs = f"{base_path}/{vehicle}/pending_images"
        pending_lbls = f"{base_path}/{vehicle}/pending_labels"
        dst_imgs = f"{base_path}/{vehicle}/round_{round_idx}/images"
        dst_lbls = f"{base_path}/{vehicle}/round_{round_idx}/labels"
        os.makedirs(dst_imgs, exist_ok=True)
        os.makedirs(dst_lbls, exist_ok=True)

        # Select 50 images
        all_imgs = sorted(os.listdir(pending_imgs))
        all_lbls = sorted(os.listdir(pending_lbls))
        selected_imgs = all_imgs[:max_images_per_round]
        selected_lbls = all_lbls[:max_images_per_round]

        for f in selected_imgs:
            shutil.move(os.path.join(pending_imgs, f), os.path.join(dst_imgs, f))
        for f in selected_lbls:
            shutil.move(os.path.join(pending_lbls, f), os.path.join(dst_lbls, f))

        num_images = len(selected_imgs)
        print(f"🔧 {vehicle}: Training on {num_images} images (remaining: {len(all_imgs) - num_images})")

        # Skip if not enough images
        if num_images < max_images_per_round:
            continue

        # Train YOLO locally
        data_yaml = update_dataset_yaml(vehicle, round_idx)
        model = YOLO(base_model_path)
        model.model.load_state_dict(global_state, strict=False)
        model.train(data=data_yaml, epochs=local_epochs, imgsz=imgsz, batch=batch,
                    project='fedprox', name=f"{vehicle}_round{round_idx}", exist_ok=True,
                    mosaic=0.9, mixup=0.2, lr0=0.001)

        # Save local model
        os.makedirs("fedprox", exist_ok=True)
        out_path = f"fedprox/{vehicle}_best_model_round{round_idx}.pt"
        torch.save({'state_dict': model.model.state_dict(), 'num_imgs': num_images, 'round': round_idx}, out_path)
        local_infos.append({'vehicle': vehicle, 'path': out_path, 'num_imgs': num_images})
        print(f"✅ {vehicle} model saved (imgs={num_images})")

    if not local_infos:
        empty_rounds += 1
        if empty_rounds >= 2:
            break
        round_idx += 1
        time.sleep(sleep_if_empty)
        continue
    empty_rounds = 0

    # =========================
    # AGGREGATION
    # =========================
    print("🔗 Aggregating client states...")
    local_states, local_names, local_weights = [], [], []
    total_imgs = sum(info['num_imgs'] for info in local_infos)
    for info in local_infos:
        ckpt = torch.load(info['path'], map_location='cpu')
        local_states.append(ckpt['state_dict'])
        local_names.append(info['vehicle'])
        local_weights.append(info['num_imgs'] / total_imgs)

    debug_states(global_state, local_states, local_names)

    agg_state = {k: v.clone() for k, v in global_state.items()}
    for k in agg_state.keys():
        tensors, wts = [], []
        for i, s in enumerate(local_states):
            if k in s and s[k].shape == agg_state[k].shape and is_finite(s[k]):
                updated = safe_fedprox_update(s[k].clone(), global_state[k].clone())
                if is_head_key(k):
                    updated = (1.0 - head_alpha) * agg_state[k] + head_alpha * updated
                tensors.append(updated)
                wts.append(local_weights[i])
        if tensors:
            try:
                agg_state[k] = weighted_avg(tensors, wts)
            except Exception:
                pass
        if not is_finite(agg_state[k]):
            agg_state[k] = torch.clamp(torch.nan_to_num(agg_state[k], nan=0.0, posinf=1e3, neginf=-1e3), -clip_max, clip_max)

    # Save aggregated global model
    global_model = YOLO(base_model_path)
    global_model.model.load_state_dict(agg_state, strict=False)
    os.makedirs(os.path.dirname(global_model_path), exist_ok=True)
    torch.save({'model': global_model.model, 'state_dict': agg_state, 'round': round_idx}, global_model_path)
    print(f"✅ Global model saved at {global_model_path} (round {round_idx})")

    # Next round
    round_idx += 1
    time.sleep(sleep_between_rounds)

print("🎉 Federated training finished.")
