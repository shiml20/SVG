import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_loss_from_log(log_paths):
    """从多个日志文件中提取并合并损失值
    
    参数:
        log_paths (str/list): 单个日志文件路径或路径列表
    
    返回:
        list: 合并后的损失值列表
    """
    if isinstance(log_paths, str):
        log_paths = [log_paths]
        
    loss = []
    for path in log_paths:
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            if " mse : " in line:
                mse = float(line.split(" mse : ")[1][:6])
                if len(line.split(" vb : ")) > 1:
                    vb = float(line.split(" vb : ")[1][:6])
                    loss.append(mse + vb)
                else:
                    loss.append(mse)
            elif "loss" in line:
                line_part = line.split("  loss : ")
                loss.append(float(line_part[1][:6]))
            elif "Train Loss:" in line:
                line_part = line.split("Train Loss: ")
                loss.append(float(line_part[1][:6]))
    # print(loss)
    return loss


# def get_loss_from_log(log_path):
#     """
#     从日志文件中提取损失值

#     参数:
#         log_path (str): 日志文件路径

#     返回:
#         list: 损失值列表
#     """
#     with open(log_path) as f:
#         lines = f.readlines()
#     loss = []

#     for line in lines:
#         if " mse : " in line:
#             mse = float(line.split(" mse : ")[1][:6])
#             if len(line.split(" vb : ")) > 1:
#                 vb = float(line.split(" vb : ")[1][:6])
#                 loss.append(mse + vb)
#             else:
#                 loss.append(mse)
#         elif "loss" in line:
#             line = line.split("  loss : ")
#             loss.append(float(line[1][:6]))
#         elif "Train Loss:" in line:
#             line = line.split("Train Loss: ")
#             loss.append(float(line[1][:6]))

#     return loss

def calculate_ema(values, span):
    """
    计算指数移动平均（EMA）

    参数:
        values (list): 原始值列表
        span (int): 平滑窗口大小

    返回:
        list: EMA 值列表
    """
    ema = [values[0]]  # 第一个EMA值就是数组的第一个值
    for i in range(1, len(values)):
        ema.append((values[i] / (span + 1)) + (1 - 1 / (span + 1)) * ema[i - 1])
    return ema


def plot_loss_comparison(log_dir_list, start_idx=100, diff_len=50000, save_path=None, 
                        x_unit=1.0, x_label="Steps", ema_span=100, mode='diff'):
    """
    绘制损失对比图

    参数:
        log_dir_list (list): 日志目录列表
        start_idx (int): 起始索引
        diff_len (int): 对比长度
        save_path (str): 如果提供，图表将保存到该路径
        x_unit (float): x 轴的缩放比例（例如，1.0 表示步数，0.001 表示千步）
        x_label (str): x 轴的标签（例如，"Steps" 或 "Time (k steps)"）
    """
    # 设置全局字体和样式
    plt.rcParams['font.family'] = 'serif'  # 使用默认衬线字体
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.dpi'] = 300


    # 创建图表
    plt.figure(figsize=(12, 7))  # 适当增加画布尺寸

    # 先处理基准模型
    base_log_dir = log_dir_list[0]
    if isinstance(base_log_dir, list):
        # base_name = "+".join([os.path.basename(d)[:20] for d in base_log_dir])
        base_name = os.path.basename(base_log_dir[0])[:300] 
        base_paths = [os.path.join(d, "log.txt") for d in base_log_dir]
    else:
        base_name = os.path.basename(base_log_dir)[:100]
        base_paths = [os.path.join(base_log_dir, "log.txt")]
    
    base_loss = np.array(get_loss_from_log(base_paths))
    
    # 遍历比较模型
    min_len = 1e9
    for i, log_dir in enumerate(tqdm(log_dir_list[1:], desc="Processing Models")):  # 跳过第一个基准模型
        # 处理日志路径
        if isinstance(log_dir, list):
            # name = "+".join([os.path.basename(d)[:20] for d in log_dir])
            name = os.path.basename(log_dir[0])[:300] 
            log_paths = [os.path.join(d, "log.txt") for d in log_dir]
        else:
            name = os.path.basename(log_dir)[:300]
            log_paths = [os.path.join(log_dir, "log.txt")]
        
        # 获取损失数据
        comp_loss = np.array(get_loss_from_log(log_paths))
        min_len = min(min(len(base_loss), len(comp_loss)), diff_len, min_len)
        print(min_len)

    for i, log_dir in enumerate(tqdm(log_dir_list[1:], desc="Processing Models")):  # 跳过第一个基准模型
        # 处理日志路径
        if isinstance(log_dir, list):
            # name = "+".join([os.path.basename(d)[:20] for d in log_dir])
            name = os.path.basename(log_dir[0])[:300] 
            log_paths = [os.path.join(d, "log.txt") for d in log_dir]
        else:
            name = os.path.basename(log_dir)[:300]
            log_paths = [os.path.join(log_dir, "log.txt")]
        
        # 获取损失数据
        comp_loss = np.array(get_loss_from_log(log_paths))
        min_len = min(min(len(base_loss), len(comp_loss)), diff_len, min_len)
        
        # 计算差异
        if mode == 'diff':
            valid_base = base_loss[start_idx:min_len] 
            valid_comp = comp_loss[start_idx:min_len]
        else:
            valid_base = base_loss[start_idx:min_len] * 0
            valid_comp = comp_loss[start_idx:min_len]
        
        diff = valid_comp - valid_base
        
        # 计算EMA
        if ema_span > 0:
            ema_values = calculate_ema(diff, ema_span)
        else:
            ema_values = diff
        
        # 生成颜色
        color = plt.cm.tab10(i % 10)
        
        # 创建x轴值
        x_values = np.arange(start_idx, min_len) * x_unit
        
        # 绘制原始差异（背景）
        plt.plot(x_values[:len(diff)], diff, 
                color=color, alpha=0.15, linewidth=0.8, zorder=1)
        
        # 绘制EMA曲线（前景）
        plt.plot(x_values[:len(ema_values)], ema_values, 
                color=color, alpha=1, linewidth=2, 
                label=name, zorder=2)

    # 添加参考线
    # plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.8, zorder=0)
    
    # 添加图例
    plt.legend(loc='upper right', frameon=False)
    # 添加标题和标签
    plt.title('Loss Comparison', fontweight='bold')
    plt.xlabel(x_label, fontweight='bold')  # 使用用户指定的 x 轴标签
    plt.ylabel('Loss Difference', fontweight='bold')

    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.6)

    # 调整布局
    plt.tight_layout()

    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {save_path}")
    else:
        plt.show()



if __name__ == "__main__":
    # 单个实验（自动合并两个断点日志）
    merged_experiment = [
    ]
    
    # 对比实验列表（可以混合单个实验和合并实验）
    log_dir_list = [
        "/ytech_m2v3_hdd/yuanziyang/sml/DiffMoE_research_local/exps/0004-3000_Dense_S_Flow-GPU4",
        "/ytech_m2v3_hdd/yuanziyang/sml/DiffMoE_research_local/exps/0004-3000_Dense_S_Flow-GPU4",
        # "/ytech_m2v3_hdd/yuanziyang/sml/DiffMoE_research_local/exps/0033-N017_TCDiT_S_AdaLN_E128C2G2_interleave_alpha1em2-GPU4",
        # "/ytech_m2v3_hdd/yuanziyang/sml/DiffMoE_research_local/exps/0034-N018_TCDiT_S_AdaLN_E64C2G2_interleave_alpha1em2-GPU4",
        "/ytech_m2v3_hdd/yuanziyang/sml/DiffMoE_research_local/exps/0008-D001_DiffMoE_S_AdaLN_E16C1-GPU4",
        "/ytech_m2v3_hdd/yuanziyang/sml/DiffMoE_research_local/exps/0024-N007_TCDiT_S_AdaLN_E32C2G2_interleave_alpha1em2-GPU4",
        # "/ytech_m2v3_hdd/yuanziyang/sml/DiffMoE_research_local/exps/0023-N006_TCDiT_S_AdaLN_E32C1G1_interleave_alpha1em2-GPU4",
        # "/ytech_m2v3_hdd/yuanziyang/sml/DiffMoE_research_local/exps/0031-N016_TCDiT_S_AdaLN_E64C1G1_interleave_alpha1em2-GPU4",
        # "/ytech_m2v3_hdd/yuanziyang/sml/DiffMoE_research_local/exps/0032-N015_TCDiT_S_AdaLN_E128C1G1_interleave_alpha1em2-GPU4",
    ]

    plot_loss_comparison(log_dir_list, save_path='[Analysis]-loss_comparison.png', start_idx=100, diff_len=10000, x_unit=0.1, x_label="Time (k steps)", ema_span=50, mode='ori')