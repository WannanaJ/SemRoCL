import os
import yaml
from datetime import datetime

def get_next_experiment_id(base_dir="experiments/stage2"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("exp_")]
    if not existing:
        return "exp_001"
    existing.sort()
    last = existing[-1]
    next_id = int(last.split("_")[1]) + 1
    return f"exp_{next_id:03d}"

def create_exp_dir(exp_id, base_dir="experiments/stage2"):
    path = os.path.join(base_dir, exp_id)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "curves"), exist_ok=True)
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    return path

def save_config(config, path):
    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    exp = get_next_experiment_id()
    p = create_exp_dir(exp)
    print(f" Created experiment folder: {p}")
