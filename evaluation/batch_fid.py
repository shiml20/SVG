import os
import subprocess

# 指定文件夹路径
folder_path = '/m2v_intern/shiminglei/DiT_MoE_Dynamic/sml_samples/'



def get_fid(folder_path):
    # 获取文件夹中所有 .npz 文件
    npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

    npz_files.sort()

    npz_files = npz_files[::-1]
    # 遍历每个 .npz 文件并调用 fid.py
    for npz_file in npz_files:
        # 构建文件的绝对路径
        file_path = os.path.join(folder_path, npz_file)

        # 调用 fid.py 并传递文件路径作为参数
        result = subprocess.run(['python', 'fid.py', file_path], capture_output=True, text=True)

        # 输出 FID 值
        print(f"File: {npz_file}")
        print(f"{result.stdout.strip()}")

# folder_path = '/m2v_intern/shiminglei/DiT_MoE_Dynamic/sml_samples/SOTA_DiffMoE-L-E8'
folder_path = '/ytech_m2v3_hdd/yuanziyang/sml/DiffMoE_research_local/sml_samples'
folder_path = '/ytech_m2v2_hdd/sml/DiffMoE_backup/sml_samples/SOTA_3000K'
folder_path = '/ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples/shift'
# folder_path = '/ytech_m2v3_hdd/yuanziyang/sml/FVG/sml_samples/cfg'
get_fid(folder_path)

# folder_path = '/m2v_intern/shiminglei/DiT_MoE_Dynamic/sml_samples/SOTA_7000K'
# get_fid(folder_path)
