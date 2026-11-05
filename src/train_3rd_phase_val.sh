#!/bin/bash
# 带验证集的微调脚本（不改动 train_3rd_phase.sh，保留原始实验可复现性）
#
# 与 train_3rd_phase.sh 的区别：
#   1. 去掉 --no-val，启用验证集 → 有真实 val loss，能早停、能选最优 epoch
#   2. 用 --val-games game_1 按视频留出验证集 → 避免同一视频相邻帧泄漏
#   3. 学习率降到 1e-5 → 只做微调，不破坏已收敛的权重
#
# 为什么验证集选 game_1：五个训练视频里只有 game_1 的三类事件分布均衡
#   game_1: empty 227 / bounce 182 / net 186  ← 均衡
#   game_5: empty 239 / bounce 179 / net   7  ← net 几乎为 0，不能用作验证
#
# 用法：
#   bash train_3rd_phase_val.sh                    # 默认 30 epoch
#   bash train_3rd_phase_val.sh --num_epochs 10    # 覆盖任意参数

python main.py \
  --working-dir '../' \
  --saved_fn 'ttnet_3rd_phase_val' \
  --val-games game_1 \
  --batch_size 8 \
  --num_workers 4 \
  --num_epochs 30 \
  --lr 0.00001 \
  --lr_type 'step_lr' \
  --lr_step_size 10 \
  --lr_factor 0.2 \
  --gpu_idx 0 \
  --global_weight 1. \
  --seg_weight 1. \
  --event_weight 1. \
  --local_weight 1. \
  --pretrained_path ../checkpoints/ttnet_3rd_phase/ttnet_3rd_phase_epoch_30.pth \
  --smooth-labelling \
  "$@"
