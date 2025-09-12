import json
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置参数
input_json_path = "/ytech_m2v2_hdd/sml/PixArt-sigma/output.json"  # 输入 JSON 文件路径
output_image_dir = "/ytech_m2v2_hdd/sml/imgs_data/resized_images"  # 输出图片目录
output_json_path = "/ytech_m2v2_hdd/sml/resized_output.json"  # 输出 JSON 文件路径
resize_size = (256, 256)  # 调整后的图片大小
max_workers = 32  # 最大线程数

# 确保输出目录存在
os.makedirs(output_image_dir, exist_ok=True)

# 加载输入 JSON 文件
with open(input_json_path, "r") as f:
    data = json.load(f)

# 处理单张图片的函数
def process_image(item):
    # 获取原始路径
    original_path = item["path"]
    # 生成新路径
    original_filename = Path(original_path).name  # 获取文件名
    new_path = os.path.join(output_image_dir, original_filename)  # 新路径

    # 打开图片并调整大小
    try:
        with Image.open(original_path) as img:
            img = img.resize(resize_size, Image.BICUBIC)  # 调整大小
            img.save(new_path)  # 保存到新路径
        # 更新 JSON 数据中的路径
        item["path"] = new_path
        return item, None  # 返回成功结果
    except Exception as e:
        return item, str(e)  # 返回错误信息

# 使用多线程处理图片
new_data = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 提交所有任务
    futures = [executor.submit(process_image, item) for item in data]
    # 使用 tqdm 显示进度条
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
        item, error = future.result()
        if error:
            print(f"Error processing {item['path']}: {error}")
        else:
            new_data.append(item)

# 保存新的 JSON 文件
with open(output_json_path, "w") as f:
    json.dump(new_data, f, indent=4)

print(f"Resized images saved to {output_image_dir}")
print(f"New JSON file saved to {output_json_path}")