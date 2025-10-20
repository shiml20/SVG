# **SVG: Latent Diffusion Model without Variational Autoencoder**

<sub>Official PyTorch Implementation</sub>

---

<div align="center">
<img src="figs/logo.svg" width="35%"/>

### [<a href="https://arxiv.org/abs/2510.15301" target="_blank">arXiv</a>] | [<a href="https://howlin-wang.github.io/svg/" target="_blank">Project Page</a>]

***[Minglei Shi<sup>1*</sup>](https://github.com/shiml20), [Haolin Wang<sup>1*</sup>](https://howlin-wang.github.io), [Wenzhao Zheng<sup>1†</sup>](https://wzzheng.net), [Ziyang Yuan<sup>2</sup>](https://scholar.google.ru/citations?user=fWxWEzsAAAAJ&hl=en), [Xiaoshi Wu<sup>2</sup>](https://scholar.google.com/citations?user=cnOAMbUAAAAJ&hl=en), [Xintao Wang<sup>2</sup>](https://xinntao.github.io), [Pengfei Wan<sup>2</sup>](https://scholar.google.com/citations?user=P6MraaYAAAAJ&hl=en), [Jie Zhou<sup>1</sup>](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en), [Jiwen Lu<sup>1</sup>](https://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)***  
<small>(*equal contribution, listed in alphabetical order; †project lead)</small>  
<sup>1</sup>Department of Automation, Tsinghua University  <sup>2</sup>Kling Team, Kuaishou Technology

</div>

---

## 🧠 Overview

We introduce **SVG**, a novel latent diffusion model without variational autoencoders, which unleashes Self-supervised representations for Visual Generation.

**Key Components:**
1. **SVG Autoencoder** - Uses a frozen representation encoder with a residual branch to compensate the information loss and a learned convolutional decoder to transfer the SVG latent space to pixel space.
2. **Latent Diffusion Transformer** - Performs diffusion modeling directly on SVG latent space.

**Repository Features:**
- ✅ PyTorch implementation of **SVG Autoencoder**.
- ✅ PyTorch implementation of **Latent Diffusion Transformer**.
- ✅ End-to-end **training** and **sampling** scripts.
- ✅ Multi-GPU distributed training support.

---

## ⚙️ Installation

### 1. Create Environment
```bash
conda create -n svg python=3.10 -y
conda activate svg
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Data Preparation

### 1. Download DINOv3
```bash
git clone https://github.com/facebookresearch/dinov3.git
```
Follow the official DINOv3 repository instructions to download pre-trained checkpoints.

### 2. Prepare Dataset
- Download **ImageNet-1k**
- Update dataset paths in the configuration files

---

## 🚀 Quick Start

### 1. Configure Paths  
Before running the pipeline, update the placeholder paths in the following configuration files to match your local file/directory structure.


#### 1.1 Autoencoder Config  
File path: `autoencoder/configs/example_svg_autoencoder_vitsp.yaml`  

Modify the paths under `dinoconfig` (for DINOv3 dependencies) and `train`/`validation` (for dataset) as shown below:  
```yaml
dinoconfig:
  dinov3_location: /path/to/your/dinov3  # Path to the directory storing the DINOv3 codebase
  model_name: dinov3_vits16plus          # Fixed DINOv3 model variant (no need to change)
  weights: /path/to/your/dinov3_vits16plus_pretrain.pth  # Path to the pre-trained DINOv3 weights file
...
train:
  params:
    data_root: /path/to/your/ImageNet-1k/  # Root directory of the ImageNet-1k dataset (for training)
validation:
  params:
    data_root: /path/to/your/ImageNet-1k/  # Root directory of the ImageNet-1k dataset (for validation)
```


#### 1.2 Diffusion Config  
File path: `configs/example_SVG_XL.yaml`  

Update the `data_path` (for training data) and `encoder_config` (path to the Autoencoder config above) as follows:  
```yaml
basic:
  data_path: /path/to/your/ImageNet-1k/train_images  # Path to the "train_images" subfolder in ImageNet-1k
  encoder_config: ../autoencoder/svg/configs/example_svg_autoencoder_vitsp.yaml  # Relative/absolute path to your edited Autoencoder config
```  

> Note: Ensure the encoder_config path is valid (use an absolute path if the relative path ../ does not match your project’s folder hierarchy). Additionally, the ckpt parameter must be set to the full path of your trained decoder checkpoint file.

### 2. Train SVG Autoencoder
```bash
cd autoencoder/svg
bash run_train.sh configs/example_svg_autoencoder_vitsp.yaml
```

### 3. Train Latent Diffusion Transformer
```bash
torchrun --nnodes=1 --nproc_per_node=8 train_svg.py --config ./configs/example_SVG_XL.yaml
```

---

## 🎨 Image Generation

Generate images using a trained model:

```bash
# Update ckpt_path in sample_svg.py with your checkpoint
python sample_svg.py
```

Generated images will be saved to the current directory.

---

## 🛠️ Configuration

### Key Configuration Files:
- `autoencoder/configs/` - SVG autoencoder training configurations
- `configs/` - Diffusion transformer training configurations

### Multi-GPU Training:
Adjust `--nproc_per_node` based on your available GPUs. The example uses 8 GPUs.

---

## 📄 Citation

If you use this work in your research, please cite our paper:

```bibtex
@misc{shi2025latentdiffusionmodelvariational,
      title={Latent Diffusion Model without Variational Autoencoder}, 
      author={Minglei Shi and Haolin Wang and Wenzhao Zheng and Ziyang Yuan and Xiaoshi Wu and Xintao Wang and Pengfei Wan and Jie Zhou and Jiwen Lu},
      year={2025},
      eprint={2510.15301},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.15301}, 
}
```
---

## 🙏 Acknowledgments

This implementation builds upon several excellent open-source projects:

* [**dinov3**](https://github.com/facebookresearch/dinov3) - Dinov3 official architecture
* [**SigLIP2**](https://huggingface.co/blog/siglip2) - SigLIP2 official architecture
* [**MAE**](https://github.com/facebookresearch/mae) - MAE baseline architecture
* [**SiT**](https://github.com/willisma/sit) - Diffusion framework and training codebase
* [**VAVAE**](https://github.com/hustvl/LightningDiT/) - PyTorch convolutional decoder implementation

---

## 📧 Contact

For questions and issues, please open an issue on GitHub or contact the authors.

---

<div align="center">
<sub>Made with ❤️ by the SVG Team</sub>
</div>