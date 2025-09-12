export http_proxy=http://10.74.176.8:11080
export https_proxy=http://10.74.176.8:11080
export HTTP_PROXY=http://10.74.176.8:11080
export HTTPS_PROXY=http://10.74.176.8:11080
export no_proxy="localhost,127.0.0.1,localaddress,localdomain.com,internal,.internal,corp.kuaishou.com,.corp.kuaishou.com,test.gifshow.com,.test.gifshow.com,staging.kuaishou.com,.staging.kuaishou.com"
export NO_PROXY="localhost,127.0.0.1,localaddress,localdomain.com,internal,.internal,corp.kuaishou.com,.corp.kuaishou.com,test.gifshow.com,.test.gifshow.com,staging.kuaishou.com,.staging.kuaishou.com"




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0139-E0004_Dense_XL_Flow_Dinov3_vitb_BS256-GPU8/checkpoints/0300000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0139-E0004_Dense_XL_Flow_Dinov3_vitb_BS256-GPU8/checkpoints/0250000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0139-E0004_Dense_XL_Flow_Dinov3_vitb_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0139-E0004_Dense_XL_Flow_Dinov3_vitb_BS256-GPU8/checkpoints/0150000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0139-E0004_Dense_XL_Flow_Dinov3_vitb_BS256-GPU8/checkpoints/0100000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0139-E0004_Dense_XL_Flow_Dinov3_vitb_BS256-GPU8/checkpoints/0050000.pt \
    --tag "euler" ;\



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v2_hdd/sml/NeurIPS25/Done-Exps/0469-6002_Dense_DiT_XL_Flow/checkpoints/0800000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v2_hdd/sml/NeurIPS25/Done-Exps/0469-6002_Dense_DiT_XL_Flow/checkpoints/1200000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v2_hdd/sml/NeurIPS25/Done-Exps/0469-6002_Dense_DiT_XL_Flow/checkpoints/1400000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v2_hdd/sml/NeurIPS25/Done-Exps/0469-6002_Dense_DiT_XL_Flow/checkpoints/1600000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v2_hdd/sml/NeurIPS25/Done-Exps/0469-6002_Dense_DiT_XL_Flow/checkpoints/1800000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v2_hdd/sml/NeurIPS25/Done-Exps/0469-6002_Dense_DiT_XL_Flow/checkpoints/2000000.pt \
    --tag "euler" ;\



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 20 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 30 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.1 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.3 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.05 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.4 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 30 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 40 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 50 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 60 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 60 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 70 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 80 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 90 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 100 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 150 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 200 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 20 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 30 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 40 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 50 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 60 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 60 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 70 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 80 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 90 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 100 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 150 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 200 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0152-E0012_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0050000.pt \
    --tag "euler" --shift 0.1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0152-E0012_LightingDiT_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0050000.pt \
    --tag "euler" --shift 0.3 ;\






CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.1 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.2 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.3 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.5 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.6 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.7 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.8 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.9 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 1.3 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\





CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 250 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\
python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0155-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino_vit.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0153-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0136-E0000_Dense_XL_Flow_Dinov3_vitsp_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0150-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0150-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0300000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0150-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0400000.pt \
    --tag "euler" --shift 0.4 ;\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0150-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_dino.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.2 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0150-E0011_Dense_XL_Flow_Dinov3_vitsp_BS256_cache_qknorm-GPU8/checkpoints/0500000.pt \
    --tag "euler" --shift 0.4 ;\

python /m2v_intern/liujie/research/dpo/gpu_stress.py ;\


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 --master_port=29529 sml_sample_ddp_feature_custimizedSampler_vavae.py --image-size 256 \
    --global-seed 0 --per-proc-batch-size 50 --num-fid-samples 50000 --cfg-scale 1.0 --num-sampling-steps 10 --sample-dir /ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples \
    --ckpt /ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0156-E0014_Dense_XL_Flow_vavae_BS256-GPU8/checkpoints/0200000.pt \
    --tag "euler" --shift 0.4 ;\





