import pandas as pd
import matplotlib.pyplot as plt
import sys

log_file = sys.argv[1] if len(sys.argv) > 1 else "training_log.csv"

df = pd.read_csv(log_file)

metrics = ["G_loss", "D_loss", "Color", "Semantic", "Adversarial", "Frequency", 
           "Perceptual", "Task", "PSNR", "SSIM", "LPIPS"]

for m in metrics:
    if m in df.columns:
        plt.figure()
        df[m].plot()
        plt.title(m)
        plt.xlabel("Iterations")
        plt.ylabel(m)
        plt.savefig(f"{m}.png")
        plt.close()

print(" Metrics plots saved!")
