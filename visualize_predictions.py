#!/usr/bin/env python3
"""
预测结果可视化：加载当前 checkpoint，对验证集样本做 GT vs Pred 叠加。
生成 3x3 矩阵：3 个样本 × [GT 姿态轴, Pred 姿态轴, 残差误差图]
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import FastPoseViT
from src.datasets import SPEEDDataset
from src.camera import Camera
from src.utils import (
    bbox_relative_translation_to_translation,
    rotation_matrix_to_quaternion,
    get_apparent_orientation,
)
from other_utils.visualization import project_axes, quat2dcm

# ============ 配置 ============
CKPT_PATH = "results/vit-b-16-384-speedplus/model.pth"
DATASET_ROOT = "SPEED_PLUS_FIXED/"
OUT_DIR = "results/vit-b-16-384-speedplus/figures"
NUM_SAMPLES = 3
IMG_SIZE = (384, 384)
os.makedirs(OUT_DIR, exist_ok=True)


def load_dataset():
    """加载验证集（与训练一致的配置：bbox crop 10%, no pixel aug）"""
    class Args:
        no_pixel_augmentation = True
        no_spatial_augmentation = True
        domain_gap_pixel_augmentation = False
        do_style_aug = False
        no_rotation_compensation = False
        use_absolute_translation = False
        use_full_image = False
        no_bbox_augmentation = True
        bbox_crop_percent = 10.0
        no_crop_padding = False
        convert_to_grayscale = False

    ds = SPEEDDataset(
        dataset_root_dir=DATASET_ROOT,
        split="val",
        rotation_format="matrix",
        img_size=IMG_SIZE,
        bbox_json_path=os.path.join(DATASET_ROOT, "speed_plus_bbox_annotations.json"),
        args=Args(),
        dataset_name="SPEED_PLUS",
    )
    return ds


def load_model():
    """加载模型（与训练参数一致）"""
    model = FastPoseViT(
        vit_model="vit_b_16_384",
        out_dim_translation=3,
        out_dim_rotation=6,
        merge_outputs=True,
        no_mlp=True,
        nb_class_tokens=1,
        num_hidden_layers=0,
        hidden_layer_dim=0,
        use_layer_norm=False,
        dropout_rate=0.0,
        use_residual=False,
    )
    state_dict = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    # 兼容保存格式
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def axes_to_image(image, q, t, camera, color="lime", label=""):
    """将卫星坐标系三轴投影到图像上（返回用于绘制的点集）"""
    p_axes = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    points_body = np.transpose(p_axes)
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(t, 1)))
    p_cam = np.dot(pose_mat, points_body)
    p_cam = p_cam[:, p_cam[2, :] > 0]
    p_img = np.dot(camera.K, p_cam)
    p_img /= p_img[2, :]
    return p_img[:2, :]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = load_dataset()
    print(f"Val samples: {len(ds)}")

    model = load_model().to(device)
    print(f"Loaded checkpoint: {CKPT_PATH}")

    camera = Camera()

    # 随机选 3 个样本
    indices = np.random.RandomState(42).choice(len(ds), NUM_SAMPLES, replace=False)

    fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(15, 5 * NUM_SAMPLES))
    if NUM_SAMPLES == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Ground Truth", "Prediction", "Pose Error (GT vs Pred)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=10)

    for row, idx in enumerate(indices):
        image, translation_gt, rotation_gt, bbox = ds[idx]

        # 推理
        with torch.no_grad():
            x = image.unsqueeze(0).to(device)
            pred = model(x).cpu().numpy()[0]

        pred_trans_rel = pred[:3]
        pred_rot_6d = pred[3:].reshape(2, 3)  # 6D -> first two columns

        # 6D rotation -> 3x3 rotation matrix (Gram-Schmidt)
        r1 = pred_rot_6d[0]
        r2 = pred_rot_6d[1]
        r1 = r1 / (np.linalg.norm(r1) + 1e-8)
        r2 = r2 - np.dot(r1, r2) * r1
        r2 = r2 / (np.linalg.norm(r2) + 1e-8)
        r3 = np.cross(r1, r2)
        pred_rot_mat = np.stack([r1, r2, r3], axis=1)

        # 相对平移 -> 绝对平移
        trans_gt_abs = bbox_relative_translation_to_translation(
            translation_gt.unsqueeze(0), bbox.unsqueeze(0), camera
        ).squeeze(0).numpy()
        trans_pred_abs = bbox_relative_translation_to_translation(
            torch.tensor(pred_trans_rel).unsqueeze(0).float(),
            bbox.unsqueeze(0),
            camera,
        ).squeeze(0).numpy()

        # 旋转矩阵 -> 四元数（用于投影）
        q_gt = rotation_matrix_to_quaternion(rotation_gt.unsqueeze(0)).squeeze(0).numpy()
        q_pred = rotation_matrix_to_quaternion(
            torch.tensor(pred_rot_mat).unsqueeze(0).float()
        ).squeeze(0).numpy()

        # 反归一化图像用于显示
        img_np = image.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = np.clip(img_np * std + mean, 0, 1)

        # 列 0: GT
        ax = axes[row, 0]
        ax.imshow(img_np)
        pts_gt = axes_to_image(img_np, q_gt, trans_gt_abs, camera, color="lime")
        origin_gt = pts_gt[:, 0]
        colors_axes = ["#ff3b30", "#34c759", "#007aff"]  # R, G, B
        for i in range(3):
            ax.plot(
                [origin_gt[0], pts_gt[0, i + 1]],
                [origin_gt[1], pts_gt[1, i + 1]],
                color=colors_axes[i], linewidth=2.5,
            )
        ax.scatter(*origin_gt, color="white", s=40, edgecolor="black", zorder=5)
        ax.set_ylabel(f"Sample {idx}", fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # 列 1: Pred
        ax = axes[row, 1]
        ax.imshow(img_np)
        pts_pred = axes_to_image(img_np, q_pred, trans_pred_abs, camera, color="orange")
        origin_pred = pts_pred[:, 0]
        for i in range(3):
            ax.plot(
                [origin_pred[0], pts_pred[0, i + 1]],
                [origin_pred[1], pts_pred[1, i + 1]],
                color=colors_axes[i], linewidth=2.5, linestyle="--",
            )
        ax.scatter(*origin_pred, color="orange", s=40, edgecolor="black", zorder=5)
        ax.set_xticks([]); ax.set_yticks([])

        # 列 2: 误差叠加图（GT 实线 + Pred 虚线）
        ax = axes[row, 2]
        ax.imshow(img_np)
        for i in range(3):
            ax.plot(
                [origin_gt[0], pts_gt[0, i + 1]],
                [origin_gt[1], pts_gt[1, i + 1]],
                color=colors_axes[i], linewidth=2.5, label=f"GT {'XYZ'[i]}",
            )
            ax.plot(
                [origin_pred[0], pts_pred[0, i + 1]],
                [origin_pred[1], pts_pred[1, i + 1]],
                color=colors_axes[i], linewidth=2.5, linestyle="--", alpha=0.8,
            )
        # 计算角度误差
        dcm_gt = quat2dcm(q_gt)
        dcm_pred = quat2dcm(q_pred)
        rel_rot = dcm_pred @ dcm_gt.T
        angle_err = np.degrees(np.arccos(np.clip((np.trace(rel_rot) - 1) / 2, -1, 1)))
        trans_err = np.linalg.norm(trans_gt_abs - trans_pred_abs)
        ax.set_title(f"Rot Err: {angle_err:.2f}° | Trans Err: {trans_err:.2f}m",
                     fontsize=10, color="#dc2626")
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "FastPose-ViT SPEED+ Prediction Visualization (ViT-B/16-384, Epoch ~54)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "prediction_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # 同时保存训练曲线 + 预测对比的合并图
    print("\nDone.")


if __name__ == "__main__":
    main()
