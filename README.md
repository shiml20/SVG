# **SVG: Latent Diffusion Model Without Variational Autoencoder**

<sub>Official PyTorch Implementation</sub>

---

<div align="center">
<img src="figs/logo.svg" width="35%"/>

### [<a href="https://huggingface.co/papers/xxx" target="_blank">arXiv</a>] | [<a href="https://howlin-wang.github.io/svg/" target="_blank">Project Page</a>]

***[Minglei Shi<sup>1*</sup>](https://github.com/shiml20), [Haolin Wang<sup>1*</sup>](https://howlin-wang.github.io), [Wenzhao Zheng<sup>1†</sup>](https://wzzheng.net), [Ziyang Yuan<sup>2</sup>](https://scholar.google.ru/citations?user=fWxWEzsAAAAJ&hl=en), [Xiaoshi Wu<sup>2</sup>](https://scholar.google.com/citations?user=cnOAMbUAAAAJ&hl=en), [Xintao Wang<sup>2</sup>](https://xinntao.github.io), [Pengfei Wan<sup>2</sup>](https://scholar.google.com/citations?user=P6MraaYAAAAJ&hl=en), [Jie Zhou<sup>1</sup>](https://www.imem.tsinghua.edu.cn/info/1330/2128.htm), [Jiwen Lu<sup>1</sup>](https://www.au.tsinghua.edu.cn/info/1078/3156.htm)***  
<small>(*equal contribution, listed alphabetically; †project lead)</small>  
<sup>1</sup>Department of Automation, Tsinghua University  <sup>2</sup>Kling Team, Kuaishou Technology

</div>

---

## 🧠 Overview

We introduce **SVG** a novel latent diffusion model without variational autoencoders, which unleashes Self-supervised representations for Visual Generation.

**Key Components:**
1. **SVG Autoencoder** - Uses a frozen representation encoder with a residual block to compensate the information loss and a learned convolutional decoder to transfer the SVG latent space to pixel space.
2. **Latent Diffusion Transformer** - Performs diffusion modeling directly on SVG latent space.

**Repository Features:**
- ✅ PyTorch implementation of **SVG Autoencoder**
- ✅ PyTorch implementation of **Latent Diffusion Transformer**
- ✅ End-to-end **training** and **sampling** scripts
- ✅ Multi-GPU distributed training support

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

**Autoencoder Config** (`autoencoder/configs/example_svg_autoencoder_vitsp.yaml`): (you should change)
```yaml
dinoconfig:
  dinov3_location: /path/to/dinov3
  model_name: dinov3_vits16plus
  weights: /path/to/dinov3_vits16plus_pretrain.pth
train:
  params:
    data_root: /path/to/ImageNet-1k/
validation:
  params:
    data_root: /path/to/ImageNet-1k/
```

**Diffusion Config** (`configs/example_SVG_XL.yaml`): (you should change)
```yaml
basic:
  data_path: /path/to/ImageNet-1k/train_images
  encoder_config: ../autoencoder/svg/configs/example_svg_autoencoder_vitsp.yaml
```

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
@article{xxxxx,
  title={Latent Diffusion Model Without Variational Autoencoder},
  author={Shi, Minglei and Wang, Haolin and Zheng, Wenzhao and Yuan, Ziyang and Wu, Xiaoshi and Wang, Xintao and Wan, Pengfei and Zhou, Jie and Lu, Jiwen},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```
---

## 🙏 Acknowledgments

This implementation builds upon several excellent open-source projects:

* [**MAE**](https://github.com/facebookresearch/mae) - ViT decoder architecture
* [**SiT**](https://github.com/willisma/sit) - Diffusion framework and training codebase
* [**LightningDiT**](https://github.com/hustvl/LightningDiT/) - PyTorch Lightning-based DiT implementation

---

## 📧 Contact

For questions and issues, please open an issue on GitHub or contact the authors.

---

<div align="center">
<sub>Made with ❤️ by the SVG Team</sub>
</div>