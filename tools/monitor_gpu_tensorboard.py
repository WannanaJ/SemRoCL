import os
import time
import argparse
import GPUtil
from datetime import timedelta
from collections import deque
from tensorboard.backend.event_processing import event_accumulator

def get_gpu_status():
    gpus = GPUtil.getGPUs()
    if not gpus:
        return None
    gpu = gpus[0]
    return {
        "load": gpu.load * 100,
        "mem_used": gpu.memoryUsed,
        "mem_total": gpu.memoryTotal,
        "temp": gpu.temperature
    }

def get_latest_event_file(log_dir):
    """自动选择日志目录下最新的事件文件"""
    if not os.path.exists(log_dir):
        return None
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("events.out")]
    if not event_files:
        return None
    return max(event_files, key=os.path.getmtime)

def get_training_progress(event_path):
    """从 TensorBoard 事件文件中读取 step 进度"""
    if not event_path or not os.path.exists(event_path):
        return 0
    ea = event_accumulator.EventAccumulator(event_path, size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception:
        return 0
    tags = ea.Tags().get("scalars", [])
    if not tags:
        return 0
    # 优先选取 Loss 或 Train step 相关标签
    step_tags = [t for t in tags if "Loss" in t or "G_total" in t or "Train" in t]
    if not step_tags:
        return 0
    scalar_events = ea.Scalars(step_tags[0])
    if not scalar_events:
        return 0
    return scalar_events[-1].step

def format_eta(seconds):
    """格式化剩余时间"""
    return str(timedelta(seconds=int(seconds))) if seconds < 1e7 else "∞"

def monitor_training(log_dir, total_epochs, steps_per_epoch, refresh, smooth):
    total_steps = total_epochs * steps_per_epoch
    print(f"🚀 GPU TensorBoard Monitor started (refresh every {refresh}s, smoothing={smooth})...")
    print("=" * 105)
    print(f"{'Time':<8} | {'GPU Load (%)':<12} | {'Mem (MB)':<12} | {'Temp (°C)':<10} | {'Step':<10} | {'Iter/s(avg)':<14} | {'ETA':<15}")
    print("-" * 105)

    last_steps = 0
    last_time = time.time()
    event_path = get_latest_event_file(log_dir)
    smooth_queue = deque(maxlen=smooth)

    try:
        while True:
            gpu = get_gpu_status()
            if not gpu:
                print("⚠️ No GPU detected.")
                time.sleep(refresh)
                continue

            new_event_path = get_latest_event_file(log_dir)
            if new_event_path != event_path:
                event_path = new_event_path
                print(f"📁 Switched to new log file: {os.path.basename(event_path)}")

            steps_done = get_training_progress(event_path)
            now = time.time()
            step_diff = max(0, steps_done - last_steps)
            time_diff = now - last_time

            iter_per_sec = step_diff / time_diff if time_diff > 0 else 0
            smooth_queue.append(iter_per_sec)
            avg_iter = sum(smooth_queue) / len(smooth_queue) if smooth_queue else iter_per_sec

            remaining_steps = max(0, total_steps - steps_done)
            eta = format_eta(remaining_steps / (avg_iter + 1e-8))

            print(f"{time.strftime('%H:%M:%S'):<8} | "
                  f"{gpu['load']:<12.1f} | "
                  f"{gpu['mem_used']:<12.1f} | "
                  f"{gpu['temp']:<10.1f} | "
                  f"{steps_done:<10d} | "
                  f"{avg_iter:<14.2f} | "
                  f"{eta:<15}")

            last_steps = steps_done
            last_time = now
            time.sleep(refresh)

    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time GPU and training progress monitor (TensorBoard version)")
    parser.add_argument("--log_dir", type=str, default="outputs/semantic_enhancement_stage2/logs", help="Path to TensorBoard log directory")
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs")
    parser.add_argument("--steps", type=int, default=563, help="Steps per epoch")
    parser.add_argument("--refresh", type=int, default=15, help="Refresh interval in seconds")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window size for iter/s")
    args = parser.parse_args()

    monitor_training(args.log_dir, args.epochs, args.steps, args.refresh, args.smooth)
