config_path=$1

torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    main_svg_autoencoder.py \
    --base $config_path \
    --train