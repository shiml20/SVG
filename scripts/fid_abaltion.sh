export http_proxy=http://10.74.176.8:11080
export https_proxy=http://10.74.176.8:11080
export HTTP_PROXY=http://10.74.176.8:11080
export HTTPS_PROXY=http://10.74.176.8:11080
export no_proxy="localhost,127.0.0.1,localaddress,localdomain.com,internal,.internal,corp.kuaishou.com,.corp.kuaishou.com,test.gifshow.com,.test.gifshow.com,staging.kuaishou.com,.staging.kuaishou.com"
export NO_PROXY="localhost,127.0.0.1,localaddress,localdomain.com,internal,.internal,corp.kuaishou.com,.corp.kuaishou.com,test.gifshow.com,.test.gifshow.com,staging.kuaishou.com,.staging.kuaishou.com"






CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\





CUDA_VISIBLE_DEVICES=6,7  torchrun --nnodes=1 --nproc_per_node=1 --master_port=29530 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0158-E0013_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=6,7  torchrun --nnodes=1 --nproc_per_node=1 --master_port=29530 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0158-E0013_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=6,7  torchrun --nnodes=1 --nproc_per_node=1 --master_port=29530 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0158-E0013_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=6,7  torchrun --nnodes=1 --nproc_per_node=1 --master_port=29530 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0158-E0013_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\




CUDA_VISIBLE_DEVICES=4,5  torchrun --nnodes=1 --nproc_per_node=1 --master_port=29531 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=4,5  torchrun --nnodes=1 --nproc_per_node=1 --master_port=29531 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=4,5  torchrun --nnodes=1 --nproc_per_node=1 --master_port=29531 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=4,5  torchrun --nnodes=1 --nproc_per_node=1 --master_port=29531 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\





CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 5 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 5 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 10 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 10 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 15 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 15 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\





CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 5 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 5 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 10 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 10 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 15 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 15 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\






CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 2.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 2.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 5.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 5.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\






CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 20 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0150-Ablation-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 20 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0150-Ablation-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 1.0 ;\






CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0153-Ablation-E0013_Dense_XL_Flow_Dinov3_vitsp_resNonorm_BS256_cache_qknorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0153-Ablation-E0013_Dense_XL_Flow_Dinov3_vitsp_resNonorm_BS256_cache_qknorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0153-Ablation-E0013_Dense_XL_Flow_Dinov3_vitsp_resNonorm_BS256_cache_qknorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0153-Ablation-E0013_Dense_XL_Flow_Dinov3_vitsp_resNonorm_BS256_cache_qknorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0150-Ablation-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0150-Ablation-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 1.0 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0750000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0750000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0157-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknormF-GPU8/checkpoints/0750000.pt \
    --tag "euler" --shift 0.4 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0159-E0016_SOTA_Dense_XL_Flow_Dinov3_vitb_resNorm_BS256_cache_qknormT-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.15 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0150-Ablation-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0150-Ablation-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\





CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 50 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0158-E0013_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknormF-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0158-E0013_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknormF-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknorm_lr1e-4_drop01_shift04_true_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0178-E0019_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_lr1e-4_drop01_shift01_true_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0153-Ablation-E0013_Dense_XL_Flow_Dinov3_vitsp_resNonorm_BS256_cache_qknorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknorm_lr1e-4_drop01_shift04_true_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0178-E0019_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_lr1e-4_drop01_shift01_true_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0153-Ablation-E0013_Dense_XL_Flow_Dinov3_vitsp_resNonorm_BS256_cache_qknorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.4 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\





CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0160-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_load500K-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 8 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0160-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_load500K-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 5 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0160-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_load500K-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 2 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0160-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_load500K-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0178-E0019_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_lr1e-4_drop01_shift01_true_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.05 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0178-E0019_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_lr1e-4_drop01_shift01_true_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.2 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0178-E0019_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_lr1e-4_drop01_shift01_true_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.3 ;\
    

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknorm_lr1e-4_drop01_shift04_true_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknorm_lr1e-4_drop01_shift04_true_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.2 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknorm_lr1e-4_drop01_shift04_true_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknorm_lr1e-4_drop01_shift04_true_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknorm_lr1e-4_drop01_shift04_true_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.2 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_Dense_XL_Flow_Dinov3_vitsp_resNorm_BS256_cache_qknorm_lr1e-4_drop01_shift04_true_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_XL_Flow_Dinov3sp_resNorm_BS256_qknorm_shift04_true_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_XL_Flow_Dinov3sp_resNorm_BS256_qknorm_shift04_true_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.2 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_XL_Flow_Dinov3sp_resNorm_BS256_qknorm_shift04_true_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.3 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_XL_Flow_Dinov3sp_resNorm_BS256_qknorm_shift04_true_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_XL_Flow_Dinov3sp_resNorm_BS256_qknorm_shift04_true_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.5 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\


cp /ytech_m2v3_hdd/yuanziyang/sml/cache/checkpoints/  ~/.cache/torch/hub/checkpoints/




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_XL_Flow_Dinov3sp_resNorm_BS256_qknorm_shift04_true_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.08 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_XL_Flow_Dinov3sp_resNorm_BS256_qknorm_shift04_true_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.09 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0179-E0019_XL_Flow_Dinov3sp_resNorm_BS256_qknorm_shift04_true_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.12 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.08 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.09 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.07 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0100000.pt \
    --tag "euler" --shift 1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0200000.pt \
    --tag "euler" --shift 1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0400000.pt \
    --tag "euler" --shift 1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/0800000.pt \
    --tag "euler" --shift 1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/1600000.pt \
    --tag "euler" --shift 1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/3000000.pt \
    --tag "euler" --shift 1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29532 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-Dense-XL/checkpoints/4500000.pt \
    --tag "euler" --shift 1 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.11 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.12 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.13 ;\



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.14 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.2 ;\



python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0300000.pt \
    --tag "euler" --shift 0.15 ;\

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0180-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm-GPU8/checkpoints/0300000.pt \
    --tag "euler" --shift 0.15 ;\





CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
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
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\






CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0160-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_load500K-GPU8/checkpoints/0850000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.25 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0160-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_load500K-GPU8/checkpoints/0850000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.3 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0160-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_load500K-GPU8/checkpoints/0850000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.15 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0160-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm_load500K-GPU8/checkpoints/0850000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0100000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0300000.pt \
    --tag "euler" --shift 1.0 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0175-E0014_Dense_XL_Flow_vavae_BS256_freatureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 1.0 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.3 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.2 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.5 ;\

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.15 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.08 ;\





