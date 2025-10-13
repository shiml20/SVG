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
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0206-E0031_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints|constant|euler|0.15|1.25|25|0200000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0206-E0031_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints|cfg_star-1-0|euler|0.15|1.55|25|0200000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0206-E0031_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.55|10|0200000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0206-E0031_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints|cfg_star-1-2|euler|0.15|2|10|0200000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0206-E0031_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints|cfg_star-1-2|euler|0.15|3|10|0200000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0206-E0031_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.25|10|0200000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0206-E0031_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints|cfg_star-1-2|euler|0.25|1.55|10|0200000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0206-E0031_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints|cfg_star-1-2|euler|0.35|1.55|10|0200000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0206-E0031_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints|cfg_star-1-2|euler|0.45|1.55|10|0200000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0206-E0031_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.55|10|0200000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0206-E0031_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.55|50|0200000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0206-E0031_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.55|100|0200000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0206-E0031_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.55|25|0200000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0199-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load4000K-GPU8/checkpoints|constant|euler|0.15|1.25|50|3000000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0199-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load4000K-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.55|50|3000000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0199-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load4000K-GPU8/checkpoints|cfg_star-1-1|euler|0.15|1.55|50|3000000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0199-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load4000K-GPU8/checkpoints|cfg_star-1-0|euler|0.15|1.55|50|3000000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
    # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0199-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load4000K-GPU8/checkpoints|cfg_star-1-0|euler|0.15|1.55|25|3000000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0199-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load4000K-GPU8/checkpoints|cfg_star-1-1|euler|0.15|1.55|25|3000000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0199-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load4000K-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.55|25|3000000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0202-E0028_XXXL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift01_featureNorm-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.55|25|1100000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0200-E0027_XL_Flow_Dinov3sp_BS256_qknorm_shift01_featureNorm_load1000K-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.55|25|3000000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0196-E0024_B_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints|constant|euler|0.15 0.05 0.2 0.3 0.4 0.5|1.0|10|0400000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0196-E0024_B_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints|constant|euler|0.15|1.0|25|0400000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.52|25|2000000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.53|25|2000000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.54|25|2000000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.56|25|2000000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.49|25|2000000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints|cfg_star-1-2|euler|0.15|1.48|25|2000000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints|cfg_star-1-0|euler|0.15|1.47|25|2000000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints|cfg_star-1-0|euler|0.15|1.5|25|1250000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0195-G0000_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load1600K_512-GPU8/checkpoints|cfg_star-1-0|euler|0.15|1.55|25|0500000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints|constant|euler|0.15|1.0|25|1600000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
  # "/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints|cfg_star-4|euler|0.15|1.2 1.25 1.3|15 25 30 50 100 250|0550000|sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py"
)

mkdir -p "$SAMPLE_DIR"

group_num=0
for entry in "${diff_configs[@]}"; do
  group_num=$((group_num + 1))

  IFS='|' read -r CKPT_BASE_DIR CFG_MODE TAG SHIFT_STR CFG_SCALES_STR STEPS_STR CHECKPOINTS_STR MAIN_SCRIPT_OPT STRESS_SCRIPT_OPT <<< "$entry" || true

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
  read -r -a SHIFT <<< "$SHIFT_STR"

  echo "======================================================"
  echo "开始执行第 $group_num 组配置"
  echo "======================================================"

  for cfg in "${CFG_SCALES[@]}"; do
    for step in "${STEPS[@]}"; do
      for ckpt in "${CHECKPOINTS[@]}"; do
        for shift in "${SHIFT[@]}"; do
          cfg_trimmed="$(echo "$cfg" | xargs)"
          step_trimmed="$(echo "$step" | xargs)"
          ckpt_trimmed="$(echo "$ckpt" | xargs)"
          shift_trimmed="$(echo "$shift" | xargs)"
          CKPT_PATH="$CKPT_BASE_DIR/$ckpt_trimmed.pt"

          echo ""
          echo "开始运行: cfg=$cfg_trimmed, steps=$step_trimmed, shift=$shift_trimmed checkpoint=$ckpt_trimmed.pt"

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
            --cfg_mode "$CFG_MODE" \
            --shift "$shift_trimmed"


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
