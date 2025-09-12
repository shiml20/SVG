from torch.utils.data import RandomSampler
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.utils.data_sampler import AspectRatioBatchSampler

def create_dataset(config, image_size, aspect_ratio_type, max_length):
    """构建训练数据集"""
    set_data_root(config.data_root)
    
    dataset = build_dataset(
        config.data,
        resolution=image_size,
        aspect_ratio_type=aspect_ratio_type,
        real_prompt_ratio=config.real_prompt_ratio,
        max_length=max_length,
        config=config
    )
    return dataset

def create_dataloader(dataset, config, train_batch_size, num_workers):
    """构建数据加载器"""
    if config.multi_scale:
        batch_sampler = AspectRatioBatchSampler(
            sampler=RandomSampler(dataset),
            dataset=dataset,
            batch_size=train_batch_size,
            aspect_ratios=dataset.aspect_ratio,
            drop_last=True,
            ratio_nums=dataset.ratio_nums,
            config=config,
            valid_num=config.valid_num
        )
        return build_dataloader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers
        )
    else:
        return build_dataloader(
            dataset,
            num_workers=num_workers,
            batch_size=train_batch_size,
            shuffle=True
        )

def prepare_data_components(config, image_size):
    """数据准备完整流程"""
    # 设置数据参数
    aspect_ratio_type = config.aspect_ratio_type if hasattr(config, 'aspect_ratio_type') else 'origin'
    max_length = config.model_max_length
    
    # 构建数据集和数据加载器
    dataset = create_dataset(
        config=config,
        image_size=image_size,
        aspect_ratio_type=aspect_ratio_type,
        max_length=max_length
    )
    
    dataloader = create_dataloader(
        dataset=dataset,
        config=config,
        train_batch_size=config.train_batch_size,
        num_workers=config.num_workers
    )
    
    return dataset, dataloader