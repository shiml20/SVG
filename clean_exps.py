import os
import torch
import glob

def process_checkpoints(root_dir):
    """
    处理所有子文件夹中的checkpoint文件，只保留ema键值
    """
    # 找到所有checkpoints文件夹
    checkpoint_dirs = glob.glob(os.path.join(root_dir, '*', 'checkpoints'))
    checkpoint_dirs = glob.glob(os.path.join(root_dir, 'checkpoints'))
    
    print(f"找到 {len(checkpoint_dirs)} 个checkpoints文件夹")
    
    for checkpoint_dir in checkpoint_dirs:
        print(f"\n处理文件夹: {checkpoint_dir}")
        
        # 找到所有的.pt文件
        pt_files = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
        
        print(f"找到 {len(pt_files)} 个.pt文件")
        
        for pt_file in pt_files:
            try:
                # 加载checkpoint
                checkpoint = torch.load(pt_file, map_location='cpu', weights_only=False)
                
                # 检查是否包含ema键
                if 'ema' in checkpoint:
                    # 创建只包含ema的新checkpoint
                    new_checkpoint = {'ema': checkpoint['ema']}
                    
                    # 保存新的checkpoint（覆盖原文件）
                    torch.save(new_checkpoint, pt_file)
                    print(f"✓ 已处理: {os.path.basename(pt_file)}")
                else:
                    print(f"⚠ 跳过（无ema键）: {os.path.basename(pt_file)}")
                    
            except Exception as e:
                print(f"✗ 处理失败 {os.path.basename(pt_file)}: {str(e)}")

def clean(root_dir='/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/000-SiT-B'):
    if not os.path.exists(root_dir):
        print(f"错误: 目录不存在: {root_dir}")
        return
    
    print("开始处理checkpoint文件...")
    print(f"根目录: {root_dir}")
    
    # 确认操作
    # response = input("确定要继续吗？这将修改原始文件 (y/n): ")
    # if response.lower() != 'y':
        # print("操作已取消")
        # return
    
    process_checkpoints(root_dir)
    print("\n处理完成！")

if __name__ == "__main__":
    clean(root_dir="/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0188-E0019_XL_Flow_Dinov3sp_resNormEpoch40_BS256_qknorm_shift04_featureNorm_load900K-GPU8/")
    # clean(root_dir="/ytech_m2v3_hdd/yuanziyang/sml/FVG/exps/0000-SiT-XL")