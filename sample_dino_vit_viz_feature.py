import os
import glob
from typing import List, Tuple, Dict
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import cv2
import torchvision.transforms as transforms

# -------------------------
# 1. 批量加载隐藏状态文件
# -------------------------
def load_hidden_states_grouped_aligned(
    folder_path: str,
    file_pattern: str = "*.pt",
    expected_shape: Tuple[int, int, int] = (1, 256, 1152)
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    批量加载隐藏状态，并按 class 和 timestep 分组，同时生成平铺列表与 info 对齐
    Returns:
        all_data_aligned: list of np.ndarray [H,C]
        file_info_aligned: list of dict 对应每个 feature
    """
    file_paths = glob.glob(os.path.join(folder_path, file_pattern))
    print(f"找到 {len(file_paths)} 个隐藏状态文件")
    
    def parse_filename(filename):
        basename = os.path.basename(filename)
        parts = basename.replace('.pt','').split('_')
        info = {}
        for part in parts:
            if part.startswith('t'):
                info['timestep'] = str(float(part[1:]))
            elif part.startswith('c'):
                info['class'] = str(int(part[1:]))
            elif part.startswith('layer'):
                info['layer'] = str(int(part[5:]))
        return info
    
    features_by_class: Dict[str, Dict[str, List[np.ndarray]]] = {}
    file_info_list: List[dict] = []
    
    # 首先构建 features_by_class 并收集 info
    for file_path in tqdm(file_paths, desc="加载文件"):
        try:
            hidden_state = torch.load(file_path)
            if hidden_state.shape != expected_shape:
                print(f"文件 {file_path} 形状异常: {hidden_state.shape}")
                continue
            feature_np = hidden_state.squeeze(0).cpu().numpy()
            
            info = parse_filename(file_path)
            info['file_path'] = file_path
            file_info_list.append(info)
            
            cls_id = info.get('class')
            timestep = info.get('timestep')
            if cls_id not in features_by_class:
                features_by_class[cls_id] = {}
            if timestep not in features_by_class[cls_id]:
                features_by_class[cls_id][timestep] = []
            features_by_class[cls_id][timestep].append(feature_np)
        except Exception as e:
            print(f"加载文件 {file_path} 出错: {e}")
    
    # 平铺 features 列表，并对齐 file_info
    all_data_aligned = []
    file_info_aligned = []
    
    for cls_id, timestep_dict in features_by_class.items():
        for timestep, feature_list in timestep_dict.items():
            for feature in feature_list:
                all_data_aligned.append(feature)
                # 匹配 file_info
                matching_info = next(
                    (info for info in file_info_list if str(info.get('class'))==cls_id and str(info.get('timestep'))==timestep),
                    {'class': cls_id, 'timestep': timestep, 'layer':'?', 'file_path':'?'}
                )
                file_info_aligned.append(matching_info)
    
    print(f"整理后总共 {len(all_data_aligned)} 个特征，与 file_info 对齐")
    return all_data_aligned, file_info_aligned

# -------------------------
# 2. 计算特征图尺寸与 patch 映射
# -------------------------
def calculate_feature_map_dimensions(image_w:int, image_h:int, patch_size:int):
    feat_w = image_w // patch_size
    feat_h = image_h // patch_size
    patch_coords = [[(x*patch_size, y*patch_size, (x+1)*patch_size, (y+1)*patch_size) 
                     for x in range(feat_w)] for y in range(feat_h)]
    return feat_w, feat_h, patch_coords

# -------------------------
# 3. 精确特征可视化（横向排列）
# -------------------------
def visualize_features_precise(all_data: List[np.ndarray],
                               file_info: List[dict],
                               original_image: np.ndarray,
                               selected_indices: List[int] = None,
                               patch_size: int = 16,
                               title: str = "Precise Feature Alignment",
                               tag: str = ""):
    
    image_h, image_w = original_image.shape[:2]
    feat_w, feat_h, patch_coords = calculate_feature_map_dimensions(image_w, image_h, patch_size)

    if selected_indices is None:
        selected_indices = list(range(len(all_data)))

    model_names = [f"Layer{file_info[idx].get('layer','?')}" for idx in selected_indices]

    fig, axes = plt.subplots(1, len(selected_indices)+1, figsize=(5*(len(selected_indices)+1),6))
    if len(selected_indices)==1:
        axes = [axes]
    
    # 显示原图
    axes[0].imshow(original_image)
    axes[0].axis('off')
    axes[0].set_xlabel(f"Class {file_info[selected_indices[0]].get('class','?')}", fontsize=18, fontweight='bold')

    for i, idx in enumerate(selected_indices):
        features = all_data[idx]
        expected_num_features = feat_w*feat_h
        if features.shape[0] != expected_num_features:
            if features.shape[0] == expected_num_features+1:
                features = features[1:]
            else:
                features = features[:expected_num_features]
                print(f"警告: {model_names[i]} 特征数量不匹配，已截断到 {expected_num_features}")
        
        # PCA -> 3 维
        features_pca = PCA(n_components=3).fit_transform(features)
        features_pca = (features_pca - features_pca.min(axis=0)) / (features_pca.max(axis=0)-features_pca.min(axis=0)+1e-8)

        # 构建特征图
        feature_map = np.zeros((image_h, image_w, 3), dtype=np.float32)
        for y in range(feat_h):
            for x in range(feat_w):
                idx_patch = y*feat_w+x
                if idx_patch < len(features_pca):
                    sx, sy, ex, ey = patch_coords[y][x]
                    feature_map[sy:ey, sx:ex] = features_pca[idx_patch]
        
        axes[i+1].imshow(feature_map)
        axes[i+1].axis('off')
        axes[i+1].set_xlabel(f"{model_names[i]} Features", fontsize=18, fontweight='bold')

        # 可选保存 overlay
        overlay = cv2.addWeighted(original_image.astype(np.float32)/255, 0.5, feature_map, 0.5, 0)
        overlay_img = (overlay*255).astype(np.uint8)
        # Image.fromarray(overlay_img).save(f"{model_names[i].lower()}_overlay.png")

    plt.tight_layout()
    plt.savefig(f"precise_feature_comparison_{tag}.png", dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------
# 4. 图像加载与尺寸调整
# -------------------------
def load_image(image_path:str, target_size:Tuple[int,int]=(256,256)):
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size
    ratio = min(target_size[0]/orig_w, target_size[1]/orig_h)
    new_w, new_h = int(orig_w*ratio), int(orig_h*ratio)
    # 调整为16倍数
    new_w = (new_w//16)*16
    new_h = (new_h//16)*16

    transform = transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)
    original_image = np.array(image.resize((new_w, new_h)))
    return image_tensor, original_image, new_w, new_h

# -------------------------
# 6. 按 class/timestep 选择特征索引
# -------------------------
def select_layers_by_class_timestep(file_info: List[dict],
                                    class_id: str,
                                    timestep: str,
                                    layers_to_show: List[int] = None) -> List[int]:
    """
    选择指定 class 和 timestep 的特征索引，并可指定显示的 layer
    Args:
        file_info: 平铺后的 file_info
        class_id: str, 目标 class
        timestep: str, 目标 timestep
        layers_to_show: list[int], 只显示这些 layer 的索引（按顺序），默认 None 显示全部
    Returns:
        selected_indices: list[int]
    """
    # 找到 class 和 timestep 对应的所有索引
    indices = [
        idx for idx, info in enumerate(file_info)
        if str(info.get('class')) == str(class_id) and str(info.get('timestep')) == str(timestep)
    ]
    print(f"找到 {len(indices)} 个 class={class_id}, timestep={timestep} 的特征")
    
    # 如果指定了 layers_to_show，只保留这些索引
    if layers_to_show is not None:
        selected_indices = [indices[i] for i in layers_to_show if i < len(indices)]
        print(f"筛选后显示 {len(selected_indices)} 个 layer: {layers_to_show}")
    else:
        selected_indices = indices
    
    return selected_indices



# -------------------------
# 5. 主流程
# -------------------------

if __name__=="__main__":

    # -------------------------
    # 选择 class=207, timestep=0.1 并显示指定 layer
    # -------------------------
    image_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/test_sample_0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8_0500000_sample100_euler_cfg4_shift0.15.png"
    # image_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/test_sample_0182-E0020_XL_Flow_Dinov3sp_BS256_qknorm_shift1_featureNorm-GPU8_0500000_sample100_euler_cfg4_shift0.15.png"
    # image_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/selected/class0330_seed1_idx3_steps50_cfg4_shift0.15.png"
    class_id = 207


    folder_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/hidden_states/"
    # 加载隐藏状态并平铺对齐
    all_data, file_info = load_hidden_states_grouped_aligned(folder_path)

    # 加载图像
    image_tensor, original_image, image_w, image_h = load_image(image_path)
    print(f"图像尺寸调整为: {image_w}x{image_h} (确保能被16整除)")



    layers_to_show = [0, 1, 2, 3, 4, 5, 6, 13, 20, 27]
    selected_indices = select_layers_by_class_timestep(file_info, class_id=f"{class_id}", timestep="0.1", layers_to_show=layers_to_show)
    # 可视化
    visualize_features_precise(all_data, file_info, original_image, selected_indices=selected_indices, patch_size=16, tag=f"class{class_id}_t0.1_layers")

    selected_indices = select_layers_by_class_timestep(file_info, class_id=f"{class_id}", timestep="0.0", layers_to_show=layers_to_show)
    visualize_features_precise(all_data, file_info, original_image, selected_indices=selected_indices, patch_size=16, tag=f"class{class_id}_t0.0_layers")


    selected_indices = select_layers_by_class_timestep(file_info, class_id=f"{class_id}", timestep="0.3", layers_to_show=layers_to_show)
    visualize_features_precise(all_data, file_info, original_image, selected_indices=selected_indices, patch_size=16, tag=f"class{class_id}_t0.3_layers")


    selected_indices = select_layers_by_class_timestep(file_info, class_id=f"{class_id}", timestep="0.5", layers_to_show=layers_to_show)
    visualize_features_precise(all_data, file_info, original_image, selected_indices=selected_indices, patch_size=16, tag=f"class{class_id}_t0.5_layers")


    selected_indices = select_layers_by_class_timestep(file_info, class_id=f"{class_id}", timestep="0.9", layers_to_show=layers_to_show)
    visualize_features_precise(all_data, file_info, original_image, selected_indices=selected_indices, patch_size=16, tag=f"class{class_id}_t0.9_layers")


    folder_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/hidden_states_vae/"


    # 加载隐藏状态并平铺对齐
    all_data, file_info = load_hidden_states_grouped_aligned(folder_path)

    # 加载图像
    image_tensor, original_image, image_w, image_h = load_image(image_path)
    print(f"图像尺寸调整为: {image_w}x{image_h} (确保能被16整除)")


    layers_to_show = [0, 1, 2, 3, 4, 5, 6, 13, 20, 27]
    selected_indices = select_layers_by_class_timestep(file_info, class_id=f"{class_id}", timestep="0.1", layers_to_show=layers_to_show)
    # 可视化
    visualize_features_precise(all_data, file_info, original_image, selected_indices=selected_indices, patch_size=16, tag=f"class{class_id}_t0.1_layers_vae")

    selected_indices = select_layers_by_class_timestep(file_info, class_id=f"{class_id}", timestep="0.0", layers_to_show=layers_to_show)
    visualize_features_precise(all_data, file_info, original_image, selected_indices=selected_indices, patch_size=16, tag=f"class{class_id}_t0.0_layers_vae")


    selected_indices = select_layers_by_class_timestep(file_info, class_id=f"{class_id}", timestep="0.3", layers_to_show=layers_to_show)
    visualize_features_precise(all_data, file_info, original_image, selected_indices=selected_indices, patch_size=16, tag=f"class{class_id}_t0.3_layers_vae")


    selected_indices = select_layers_by_class_timestep(file_info, class_id=f"{class_id}", timestep="0.5", layers_to_show=layers_to_show)
    visualize_features_precise(all_data, file_info, original_image, selected_indices=selected_indices, patch_size=16, tag=f"class{class_id}_t0.5_layers_vae")


    selected_indices = select_layers_by_class_timestep(file_info, class_id=f"{class_id}", timestep="0.9", layers_to_show=layers_to_show)
    visualize_features_precise(all_data, file_info, original_image, selected_indices=selected_indices, patch_size=16, tag=f"class{class_id}_t0.9_layers_vae")
