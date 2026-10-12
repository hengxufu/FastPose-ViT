#!/usr/bin/env bash
# 持续监控 FastPose-ViT 训练进度与剩余时间
# 用法: bash monitor_training.sh [刷新间隔秒数，默认 10]

set -u
LOG_FILE="results/vit-b-16-384-speedplus/train.log"
REFRESH="${1:-10}"

# 总 epoch 数（从启动参数解析，默认 310）
TOTAL_EPOCHS=310
TRAIN_ITERS=5995   # 47966 / 8
VAL_ITERS=1500     # 11994 / 8

# ANSI 颜色
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'

fmt_time() {
    local s=$1
    local h=$((s/3600)); local m=$(( (s%3600)/60 )); local sec=$((s%60))
    printf "%02d:%02d:%02d" "$h" "$m" "$sec"
}

while true; do
    clear
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║         FastPose-ViT 训练监控  (GPU 3)                       ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo -e "  刷新间隔: ${REFRESH}s   |   日志: $LOG_FILE"
    echo ""

    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${RED}[!] 日志文件不存在，等待中...${NC}"
        sleep "$REFRESH"
        continue
    fi

    # 提取所有已完成 epoch 的训练耗时（去掉 epoch 1 的编译预热）
    # 格式: "Epoch N [Train]: 6000it [MM:SS"
    mapfile -t TRAIN_TIMES < <(tr '\r' '\n' < "$LOG_FILE" \
        | grep -oE "Epoch [0-9]+ \[Train\]: ${TRAIN_ITERS}it \[[0-9]+:[0-9]+" \
        | sort -u \
        | grep -v "Epoch 1 " \
        | grep -oE "\[[0-9]+:[0-9]+$" \
        | tr -d '[]')

    # 计算平均训练耗时（秒）
    avg_train=0; cnt=0
    for t in "${TRAIN_TIMES[@]}"; do
        mm="${t%:*}"; ss="${t##*:}"
        sec=$((10#$mm * 60 + 10#$ss))
        avg_train=$((avg_train + sec))
        cnt=$((cnt + 1))
    done
    if [ "$cnt" -gt 0 ]; then
        avg_train=$((avg_train / cnt))
    else
        avg_train=341  # fallback
    fi
    avg_val=38
    avg_epoch=$((avg_train + avg_val))

    # 当前进度（最后一行 Epoch 状态）
    LAST_EPOCH_LINE=$(tr '\r' '\n' < "$LOG_FILE" | grep -oE "Epoch [0-9]+ \[(Train|Val)\].*" | tail -n 1)
    CUR_EPOCH=$(echo "$LAST_EPOCH_LINE" | grep -oE "Epoch [0-9]+" | grep -oE "[0-9]+")
    CUR_PHASE=$(echo "$LAST_EPOCH_LINE" | grep -oE "\[(Train|Val)\]" | tr -d '[]')
    CUR_ITER=$(echo "$LAST_EPOCH_LINE" | grep -oE "[0-9]+/[0-9]+" | head -n1 | cut -d/ -f1)
    TOTAL_ITER=$(echo "$LAST_EPOCH_LINE" | grep -oE "[0-9]+/[0-9]+" | head -n1 | cut -d/ -f2)
    CUR_LOSS=$(echo "$LAST_EPOCH_LINE" | grep -oE "loss=[0-9.]+" | head -n1 | cut -d= -f2)
    CUR_ROT=$(echo "$LAST_EPOCH_LINE" | grep -oE "rot_loss=[0-9.]+" | head -n1 | cut -d= -f2)
    CUR_TRANS=$(echo "$LAST_EPOCH_LINE" | grep -oE "trans_loss=[0-9.]+" | head -n1 | cut -d= -f2)
    CUR_SPEED=$(echo "$LAST_EPOCH_LINE" | grep -oE "[0-9.]+it/s" | head -n1)

    CUR_EPOCH="${CUR_EPOCH:-0}"
    CUR_PHASE="${CUR_PHASE:---}"
    CUR_ITER="${CUR_ITER:-0}"
    TOTAL_ITER="${TOTAL_ITER:-0}"

    # 进度百分比
    if [ "$CUR_EPOCH" -gt 0 ] && [ "$TOTAL_ITER" -gt 0 ]; then
        iter_pct=$(( CUR_ITER * 100 / TOTAL_ITER ))
        overall=$(( ( (CUR_EPOCH - 1) * 100 + iter_pct ) / TOTAL_EPOCHS ))
    else
        iter_pct=0; overall=0
    fi

    # 剩余时间估算
    completed_epochs=$((CUR_EPOCH - 1))
    if [ "$CUR_PHASE" = "Val" ]; then
        # 当前 epoch 训练已完成，验证中
        remaining_in_epoch=$(( avg_val - ( CUR_ITER * avg_val / ( VAL_ITERS > 0 ? VAL_ITERS : 1500 ) ) ))
    else
        remaining_in_epoch=$(( avg_train - ( CUR_ITER * avg_train / TRAIN_ITERS ) ))
    fi
    remaining_full_epochs=$(( TOTAL_EPOCHS - CUR_EPOCH ))
    remaining_sec=$(( remaining_in_epoch + remaining_full_epochs * avg_epoch ))
    eta=$(fmt_time $remaining_sec)

    # 预计完成时间
    finish_ts=$(( $(date +%s) + remaining_sec ))
    finish_str=$(date -d "@$finish_ts" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -r "$finish_ts" "+%Y-%m-%d %H:%M:%S")

    echo -e "  ${GREEN}当前状态:${NC}  Epoch ${YELLOW}${CUR_EPOCH}${NC}/${TOTAL_EPOCHS}  [${CUR_PHASE}]  ${iter_pct}%  (${CUR_ITER}/${TOTAL_ITER} it)"
    echo -e "  ${GREEN}总进度:${NC}    ${overall}%"
    echo ""
    echo -e "  ${GREEN}当前 Loss:${NC}  total=${CUR_LOSS}  rot=${CUR_ROT}  trans=${CUR_TRANS}"
    echo -e "  ${GREEN}速度:${NC}      ${CUR_SPEED:---}"
    echo ""
    echo -e "  ${GREEN}平均耗时:${NC}  Train=${avg_train}s  Val=${avg_val}s  Total=${avg_epoch}s/epoch"
    echo -e "  ${GREEN}剩余轮数:${NC}  ${remaining_full_epochs} epochs + 当前轮剩余"
    echo -e "  ${GREEN}预计剩余:${NC}  ${YELLOW}${eta}${NC}"
    echo -e "  ${GREEN}预计完成:${NC}  ${finish_str}"
    echo ""

    # 最近 3 个已完成 epoch 的训练 loss（取 epoch 末尾的 loss 值）
    echo -e "${CYAN}  ── 最近 Epoch 训练 loss 概况 ──${NC}"
    tr '\r' '\n' < "$LOG_FILE" \
        | grep -oE "Epoch [0-9]+ \[Train\]: ${TRAIN_ITERS}it \[[0-9]+:[0-9]+.*loss=[0-9.]+" \
        | sort -u \
        | tail -n 3 \
        | sed -E 's/(Epoch [0-9]+ \[Train\]): .*loss=([0-9.]+)/  \1  ->  train_loss=\2/'

    echo ""
    echo -e "${CYAN}  ── 最近 Epoch 验证 loss 概况 ──${NC}"
    tr '\r' '\n' < "$LOG_FILE" \
        | grep -oE "Epoch [0-9]+ \[Val\]: ${VAL_ITERS}it \[[0-9]+:[0-9]+.*loss=[0-9.]+" \
        | sort -u \
        | tail -n 3 \
        | sed -E 's/(Epoch [0-9]+ \[Val\]): .*loss=([0-9.]+)/  \1  ->  val_loss=\2/'

    echo ""
    echo -e "  按 ${RED}Ctrl+C${NC} 停止监控"
    sleep "$REFRESH"
done
