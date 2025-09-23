export http_proxy=http://10.74.176.8:11080
export https_proxy=http://10.74.176.8:11080
export HTTP_PROXY=http://10.74.176.8:11080
export HTTPS_PROXY=http://10.74.176.8:11080
export no_proxy="localhost,127.0.0.1,localaddress,localdomain.com,internal,.internal,corp.kuaishou.com,.corp.kuaishou.com,test.gifshow.com,.test.gifshow.com,staging.kuaishou.com,.staging.kuaishou.com"
export NO_PROXY="localhost,127.0.0.1,localaddress,localdomain.com,internal,.internal,corp.kuaishou.com,.corp.kuaishou.com,test.gifshow.com,.test.gifshow.com,staging.kuaishou.com,.staging.kuaishou.com"




cp /ytech_m2v3_hdd/yuanziyang/sml/cache/checkpoints/  ~/.cache/torch/hub/checkpoints/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/0450000.pt \
    --tag "euler" --shift 0.15 --cfg_mode linear ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/0450000.pt \
    --tag "euler" --shift 0.15 --cfg_mode constant ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/0450000.pt \
    --tag "euler" --shift 0.15 --cfg_mode interval ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/0450000.pt \
    --tag "euler" --shift 0.15 --cfg_mode late ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 20 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/0450000.pt \
    --tag "euler" --shift 0.15 --cfg_mode linear ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 20 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/0450000.pt \
    --tag "euler" --shift 0.15 --cfg_mode constant ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 20 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/0450000.pt \
    --tag "euler" --shift 0.15 --cfg_mode interval ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 20 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/0450000.pt \
    --tag "euler" --shift 0.15 --cfg_mode late ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/0450000.pt \
    --tag "euler" --shift 0.15 --cfg_mode cfg_star ;\



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 20 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/0450000.pt \
    --tag "euler" --shift 0.15 --cfg_mode cfg_star ;\



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 20 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.15 --cfg_mode cfg_star-1 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 25 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.15 --cfg_mode cfg_star-1 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py --image-size 512 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.55 --num-sampling-steps 25 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0195-G0000_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load1600K_512-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.15 --cfg_mode cfg_star-1-0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py --image-size 512 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.55 --num-sampling-steps 50 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0195-G0000_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load1600K_512-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.15 --cfg_mode cfg_star-1-0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit_ablation_guidance.py --image-size 512 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 25 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0195-G0000_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load1600K_512-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.15 --cfg_mode constant ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\
