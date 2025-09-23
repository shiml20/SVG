#!/bin/bash
# set -euo pipefail   # 注释掉，这样不会因为某一步错误直接退出

# ======================== 全局固定参数（所有配置组共用） ========================
IMAGE_SIZE=256
GLOBAL_SEED=0
PER_PROC_BATCH_SIZE=50
NUM_FID_SAMPLES=50000
SAMPLE_DIR="/ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples"
MASTER_PORT=29529  # 全局统一端口号
# 全局Python主脚本路径（默认）
MAIN_PYTHON_SCRIPT="sml_sample_ddp_feature_custimizedSampler_dino_vit.py"
# 全局GPU压力测试脚本路径（默认）
GPU_STRESS_SCRIPT="/m2v_intern/liujie/research/dpo/gpu_stress.py"
# ==============================================================================

diff_configs=(
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints|euler|0.15|1.0|3 5 8 10 20 |0100000 0200000 0300000|sml_sample_ddp_feature_custimizedSampler_dino_vit.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints|euler|0.15|1.0 1.25|10 20|0500000|sml_sample_ddp_feature_custimizedSampler_dino_vit.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0183-E0021_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints|euler|0.15|1.0 1.25|5|0400000|sml_sample_ddp_feature_custimizedSampler_dino_vit.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0190-E0026_S_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints|euler|0.15|1.0 1.25|10|0100000 0200000 0300000 0400000 0500000|sml_sample_ddp_feature_custimizedSampler_dino_vit.py"
  "//ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints|euler|0.15|1.25|50|0500000|sml_sample_ddp_feature_custimizedSampler_dino_vit.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0185-E0022_L_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints|euler|0.15|1.0 1.25|10|0100000 0200000 0300000 0400000|sml_sample_ddp_feature_custimizedSampler_dino_vit.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints|euler|1|1.5|100 250|0400000|sml_sample_ddp_feature_custimizedSampler.py"
)

mkdir -p "$SAMPLE_DIR"

group_num=0
for entry in "${diff_configs[@]}"; do
  group_num=$((group_num + 1))

  IFS='|' read -r CKPT_BASE_DIR TAG SHIFT CFG_SCALES_STR STEPS_STR CHECKPOINTS_STR MAIN_SCRIPT_OPT STRESS_SCRIPT_OPT <<< "$entry" || true

  current_main_script="${MAIN_PYTHON_SCRIPT}"
  current_stress_script="${GPU_STRESS_SCRIPT}"

  if [[ -n "${MAIN_SCRIPT_OPT:-}" ]]; then
    current_main_script="${MAIN_SCRIPT_OPT}"
  fi
  if [[ -n "${STRESS_SCRIPT_OPT:-}" ]]; then
    current_stress_script="${STRESS_SCRIPT_OPT}"
  fi

  read -r -a CFG_SCALES <<< "$CFG_SCALES_STR"
  read -r -a STEPS <<< "$STEPS_STR"
  read -r -a CHECKPOINTS <<< "$CHECKPOINTS_STR"

  echo "======================================================"
  echo "开始执行第 $group_num 组配置"
  echo "======================================================"

  for cfg in "${CFG_SCALES[@]}"; do
    for step in "${STEPS[@]}"; do
      for ckpt in "${CHECKPOINTS[@]}"; do
        cfg_trimmed="$(echo "$cfg" | xargs)"
        step_trimmed="$(echo "$step" | xargs)"
        ckpt_trimmed="$(echo "$ckpt" | xargs)"
        CKPT_PATH="$CKPT_BASE_DIR/$ckpt_trimmed.pt"

        echo ""
        echo "开始运行: cfg=$cfg_trimmed, steps=$step_trimmed, checkpoint=$ckpt_trimmed.pt"

        if [[ ! -f "$CKPT_PATH" ]]; then
          echo "错误: checkpoint 文件不存在: $CKPT_PATH"
          continue   # ⚠️ 这里改为跳过，而不是 exit
        fi

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
          --nnodes=1 \
          --nproc_per_node=8 \
          --master_port="$MASTER_PORT" \
          "$current_main_script" \
          --image-size "$IMAGE_SIZE" \
          --global-seed "$GLOBAL_SEED" \
          --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
          --num-fid-samples "$NUM_FID_SAMPLES" \
          --cfg-scale "$cfg_trimmed" \
          --num-sampling-steps "$step_trimmed" \
          --sample-dir "$SAMPLE_DIR" \
          --ckpt "$CKPT_PATH" \
          --tag "$TAG" \
          --shift "$SHIFT"

        status=$?
        if [[ $status -ne 0 ]]; then
          echo "警告: 主命令执行失败 (组=$group_num, cfg=$cfg_trimmed, steps=$step_trimmed, ckpt=$ckpt_trimmed.pt)"
          # ⚠️ 不退出，继续跑下一组
        fi

        echo "完成运行: cfg=$cfg_trimmed, steps=$step_trimmed, checkpoint=$ckpt_trimmed.pt"
        echo "----------------------------------------"
      done
    done
  done

  echo "第 $group_num 组配置执行完毕"
  echo "======================================================"
  echo
done

# ⚡ 不管前面成功失败，最后统一执行一次 GPU 压力测试
if [[ -f "$GPU_STRESS_SCRIPT" ]]; then
  echo "开始运行 GPU 压力测试脚本: $GPU_STRESS_SCRIPT"
  python "$GPU_STRESS_SCRIPT" || {
    echo "警告: GPU 压力测试脚本退出非零，但脚本继续。"
  }
else
  echo "注意: GPU 压力测试脚本 '$GPU_STRESS_SCRIPT' 未找到，跳过。"
fi

echo "所有配置组任务已完成"
