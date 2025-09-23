export http_proxy=http://10.74.176.8:11080
export https_proxy=http://10.74.176.8:11080
export HTTP_PROXY=http://10.74.176.8:11080
export HTTPS_PROXY=http://10.74.176.8:11080
export no_proxy="localhost,127.0.0.1,localaddress,localdomain.com,internal,.internal,corp.kuaishou.com,.corp.kuaishou.com,test.gifshow.com,.test.gifshow.com,staging.kuaishou.com,.staging.kuaishou.com"
export NO_PROXY="localhost,127.0.0.1,localaddress,localdomain.com,internal,.internal,corp.kuaishou.com,.corp.kuaishou.com,test.gifshow.com,.test.gifshow.com,staging.kuaishou.com,.staging.kuaishou.com"




cp /ytech_m2v3_hdd/yuanziyang/sml/cache/checkpoints/  ~/.cache/torch/hub/checkpoints/




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0183-E0021_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0183-E0021_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0183-E0021_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0300000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0184-E0020_XL_Flow_Dinov3sp_resNoNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0184-E0020_XL_Flow_Dinov3sp_resNoNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0183-E0021_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0183-E0021_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0183-E0021_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0300000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0184-E0020_XL_Flow_Dinov3sp_resNoNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0184-E0020_XL_Flow_Dinov3sp_resNoNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0200000.pt \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0300000.pt \
    --tag "euler" --shift 0.15 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0300000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0300000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.15 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\





CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0183-E0021_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0183-E0021_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0184-E0020_XL_Flow_Dinov3sp_resNoNormEpoch40_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0300000.pt \
    --tag "euler" --shift 1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 1 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0550000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0300000.pt \
    --tag "euler" --shift 0.15 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0300000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.15 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.9 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.8 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.7 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.6 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.5 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.3 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.2 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.8 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.8 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.8 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.8 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0600000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 20 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0600000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0600000.pt \
    --tag "euler" --shift 0.15 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 20 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0800000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0800000.pt \
    --tag "euler" --shift 0.15 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\