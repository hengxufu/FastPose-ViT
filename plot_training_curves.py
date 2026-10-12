#!/usr/bin/env python3
"""生成训练收敛曲线可视化（从 tensorboard events 读取）。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR = "results/vit-b-16-384-speedplus/"
OUT_DIR = "results/vit-b-16-384-speedplus/figures"
os.makedirs(OUT_DIR, exist_ok=True)

ea = EventAccumulator(LOG_DIR)
ea.Reload()
tags = ea.Tags()["scalars"]

def get_series(tag):
    evs = ea.Scalars(tag)
    return [e.step for e in evs], [e.value for e in evs]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("FastPose-ViT SPEED+ Training Curves (ViT-B/16-384)", fontsize=14, fontweight="bold")

# 1. Total Loss (Train vs Val)
ax = axes[0, 0]
for label, tag, color in [("Train", "Loss/Train/Total", "#2563eb"), ("Validation", "Loss/Validation/Total", "#dc2626")]:
    s, v = get_series(tag)
    ax.plot(s, v, label=label, color=color, linewidth=1.5)
ax.set_title("Total Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)

# 2. Rotation Loss
ax = axes[0, 1]
for label, tag, color in [("Train", "Loss/Train/Rotation", "#2563eb"), ("Validation", "Loss/Validation/Rotation", "#dc2626")]:
    s, v = get_series(tag)
    ax.plot(s, v, label=label, color=color, linewidth=1.5)
ax.set_title("Rotation Loss (6D relative Frobenius)"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)

# 3. Translation Loss
ax = axes[1, 0]
for label, tag, color in [("Train", "Loss/Train/Translation", "#2563eb"), ("Validation", "Loss/Validation/Translation", "#dc2626")]:
    s, v = get_series(tag)
    ax.plot(s, v, label=label, color=color, linewidth=1.5)
ax.set_title("Translation Loss (relative)"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)

# 4. Learning Rate
ax = axes[1, 1]
s, v = get_series("LearningRate")
ax.plot(s, v, color="#059669", linewidth=1.5)
ax.set_title("Learning Rate (CosineAnnealingWarmRestarts)"); ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
ax.set_yscale("log"); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = os.path.join(OUT_DIR, "training_curves.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

# Print summary stats
for tag in ["Loss/Train/Total", "Loss/Validation/Total"]:
    s, v = get_series(tag)
    if v:
        print(f"{tag}: final={v[-1]:.5f}  min={min(v):.5f} (epoch {v.index(min(v))+1})")
