#!/usr/bin/env python3
"""生成训练收敛曲线可视化（从 tensorboard events 读取）- 中文界面"""
import os
import matplotlib
matplotlib.use("Agg")
# 中文字体支持
from matplotlib import font_manager
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(CJK_FONT_PATH):
    font_manager.fontManager.addfont(CJK_FONT_PATH)
    matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
else:
    matplotlib.rcParams['font.sans-serif'] = ['Noto Serif CJK SC', 'DejaVu Sans', 'SimHei', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
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
fig.suptitle("FastPose-ViT SPEED+ 训练曲线 (ViT-B/16-384)", fontsize=14, fontweight="bold")

# 1. 总损失 (训练 vs 验证)
ax = axes[0, 0]
for label, tag, color in [("训练集", "Loss/Train/Total", "#2563eb"), ("验证集", "Loss/Validation/Total", "#dc2626")]:
    s, v = get_series(tag)
    ax.plot(s, v, label=label, color=color, linewidth=1.5)
ax.set_title("总损失"); ax.set_xlabel("轮次 (Epoch)"); ax.set_ylabel("损失值")
ax.legend(); ax.grid(True, alpha=0.3)

# 2. 旋转损失
ax = axes[0, 1]
for label, tag, color in [("训练集", "Loss/Train/Rotation", "#2563eb"), ("验证集", "Loss/Validation/Rotation", "#dc2626")]:
    s, v = get_series(tag)
    ax.plot(s, v, label=label, color=color, linewidth=1.5)
ax.set_title("旋转损失 (6D 相对 Frobenius)"); ax.set_xlabel("轮次 (Epoch)"); ax.set_ylabel("损失值")
ax.legend(); ax.grid(True, alpha=0.3)

# 3. 平移损失
ax = axes[1, 0]
for label, tag, color in [("训练集", "Loss/Train/Translation", "#2563eb"), ("验证集", "Loss/Validation/Translation", "#dc2626")]:
    s, v = get_series(tag)
    ax.plot(s, v, label=label, color=color, linewidth=1.5)
ax.set_title("平移损失 (相对)"); ax.set_xlabel("轮次 (Epoch)"); ax.set_ylabel("损失值")
ax.legend(); ax.grid(True, alpha=0.3)

# 4. 学习率
ax = axes[1, 1]
s, v = get_series("LearningRate")
ax.plot(s, v, color="#059669", linewidth=1.5)
ax.set_title("学习率 (余弦退火重启)"); ax.set_xlabel("轮次 (Epoch)"); ax.set_ylabel("学习率")
ax.set_yscale("log"); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = os.path.join(OUT_DIR, "training_curves.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"已保存: {out_path}")

# 打印汇总统计
for tag, label in [("Loss/Train/Total", "训练总损失"), ("Loss/Validation/Total", "验证总损失")]:
    s, v = get_series(tag)
    if v:
        print(f"{label}: 最终={v[-1]:.5f}  最小={min(v):.5f} (第 {v.index(min(v))+1} 轮)")
