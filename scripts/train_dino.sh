
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0003_Dense_XL_Flow_Dinov3_vithp_BS256.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0004_Dense_XL_Flow_Dinov3_vitb_BS256.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0005_Dense_XXL_Flow_Dinov3_vithp_BS256.yaml



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0009_Dense_XL_Flow_Dinov3_vitb_BS256_cache.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0010_Dense_XXL_Flow_Dinov3_vitb_BS256_qknorm.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0009_Dense_XL_Flow_Dinov3_vitb_BS256_cacheEfficient.yaml



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0012_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0013_Dense_XL_Flow_Dinov3_vitsp_res_BS256_cache_qknorm.yaml



CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_port=12326 \
    train_vavae.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0014_Dense_XL_Flow_vavae_BS256_freatureNorm.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF.yaml



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0013_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknormF.yaml



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0015_Dense_XL_Flow_Dinov3_vitb_BS256_qknormF.yaml


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0015_SOTA_Dense_XL_Flow_Dinov3_vitb_resNorm_BS256_cache_qknormT.yaml


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_load500K.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF_load800K.yaml


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0016_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_lr1e-3.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0016_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_lr1e-4_beta2_095.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0016_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_lr1e-4_drop03.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0017_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_lr1e-4_drop01_shift01.yaml


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0017_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_lr1e-4_drop01_shift01_true.yaml

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0018_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_lr1e-4_drop01_shift01_onestep.yaml




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0019_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_lr1e-4_drop01_shift01_true_featureNorm.yaml



CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0019_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_lr1e-4_drop01_shift01_true_featureNorm.yaml


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0020_XL_Flow_Dinov3sp_resNoNormEpoch40_BS256_qknorm_shift1_featureNorm.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0022_L_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0021_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknormF_shift1_featureNorm.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0024_B_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm.yaml


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K.yaml


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0024_B_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm.yaml


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3_repa.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0024_B_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm_repa.yaml

bash scripts/batch_fid_ablation_0919.sh ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0024_B_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm.yaml ;\


bash scripts/batch_fid_ablation_0919.sh ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0026_S_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift01_featureNorm.yaml ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=12326 \
    train_dinov3.py \
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0024_B_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift01_featureNorm.yaml