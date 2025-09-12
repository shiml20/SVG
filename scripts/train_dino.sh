
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
    --config /ytech_m2v3_hdd/yuanziyang/sml/FVG/config/E0014_Dense_XL_Flow_vavae_BS256.yaml

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