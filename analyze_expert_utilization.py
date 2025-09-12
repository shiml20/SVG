import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

class ExpertUsageAnalyzer:
    def __init__(self, file_path="/ytech_m2v2_hdd/sml/DiffMoE_research_local/expert_utilization.txt"):
        self.file_path = file_path
        self.ensure_file_exists()
        
    def ensure_file_exists(self):
        """确保输出文件存在且有正确的文件头"""
        if not Path(self.file_path).exists():
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, "w") as f:
                f.write("# Expert utilization ratios (comma-separated)\n")
            print(f"Created new log file at {self.file_path}")
    
    def load_data(self):
        """加载并解析数据"""
        try:
            with open(self.file_path, "r") as f:
                lines = [line.strip() for line in f if not line.startswith("#") and line.strip()]
            
            data = []
            for line in lines:
                try:
                    values = [float(x) for x in line.split(",")]
                    data.append(values)
                except ValueError as e:
                    print(f"Warning: Skipping malformed line - {str(e)}")
                    continue
            
            return np.array(data) if data else None
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def analyze(self):
        """执行完整分析"""
        data = self.load_data()
        if data is None or len(data) == 0:
            print("No valid data available for analysis")
            return None
        
        results = {
            "data": data,
            "averages": np.mean(data, axis=0),
            "std_devs": np.std(data, axis=0),
            "min_values": np.min(data, axis=0),
            "max_values": np.max(data, axis=0),
            "median": np.median(data, axis=0)
        }
        
        self.print_results(results)
        self.save_results(results)
        self.plot_results(results)
        
        return results
    
    def print_results(self, results):
        """打印统计结果"""
        print("\n=== Expert Utilization Statistics ===")
        print(f"{'Expert':<8} {'Avg':<10} {'StdDev':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
        for i in range(len(results["averages"])):
            print(f"{i:<8} {results['averages'][i]:<10.4f} "
                  f"{results['std_devs'][i]:<10.4f} "
                  f"{results['min_values'][i]:<10.4f} "
                  f"{results['max_values'][i]:<10.4f} "
                  f"{results['median'][i]:<10.4f}")
    
    def save_results(self, results, stats_file="expert_utilization_stats.csv"):
        """保存统计结果到CSV"""
        with open(stats_file, "w") as f:
            f.write("Expert,Average,StdDev,Min,Max,Median\n")
            for i in range(len(results["averages"])):
                f.write(f"{i},{results['averages'][i]:.6f},"
                        f"{results['std_devs'][i]:.6f},"
                        f"{results['min_values'][i]:.6f},"
                        f"{results['max_values'][i]:.6f},"
                        f"{results['median'][i]:.6f}\n")
        print(f"\nStatistics saved to {stats_file}")
    
    def plot_results(self, results, plot_file="expert_utilization_plot.png"):
        """绘制结果图表"""
        plt.figure(figsize=(12, 6))
        
        # 时间序列图
        plt.subplot(1, 2, 1)
        for expert in range(results["data"].shape[1]):
            plt.plot(results["data"][:, expert], alpha=0.7, label=f"Expert {expert}")
        plt.title("Utilization Over Time")
        plt.xlabel("Step")
        plt.ylabel("Utilization Ratio")
        plt.grid(True)
        
        # 统计摘要图
        plt.subplot(1, 2, 2)
        positions = np.arange(len(results["averages"]))
        plt.bar(positions, results["averages"], yerr=results["std_devs"],
                capsize=5, alpha=0.7)
        plt.title("Average Utilization")
        plt.xlabel("Expert ID")
        plt.ylabel("Mean ± StdDev")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        print(f"Plot saved to {plot_file}")

# 使用示例
if __name__ == "__main__":
    analyzer = ExpertUsageAnalyzer()
    analysis_results = analyzer.analyze()