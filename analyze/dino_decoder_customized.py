import torch
from ldm.modules.diffusionmodules.model import Decoder

# ==== ddconfig 转换为 Python 字典 ====
ddconfig = {
    "double_z": True,
    "z_channels": 384,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 1, 2, 2, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [16],
    "dropout": 0.0,
}

device = "cuda"


REPO_DIR = "/ytech_m2v3_hdd/yuanziyang/sml/dinov3"
dinov3_vits16plus = torch.hub.load(
    REPO_DIR,
    'dinov3_vits16plus',
    source='local',
    weights="/ytech_m2v3_hdd/yuanziyang/sml/FVG/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
)

dinov3_vits16plus = dinov3_vits16plus.to(device=device)


from torchvision import transforms
from PIL import Image
# load image
# ==== load image ====
image_path = "/m2v_intern/public_datasets/ImageNet-1k/data/train_images/n02095314/n02095314_821_n02095314.JPEG"  # 改成你要测试的图片路径
image_path = "/ytech_m2v3_hdd/yuanziyang/sml/Wan2.1/examples/flf2v_input_last_frame.png"
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),  # [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
img = Image.open(image_path).convert("RGB")
x = transform(img).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]


with torch.no_grad():
    # x = vae.encode(x).latent_dist.sample().mul_(0.18215)
    # x = vae.encode(x).latent_dist.sample()
    # import ipdb; ipdb.set_trace()
    z = dinov3_vits16plus(x)[1]["x_norm_patchtokens"]

z = z.reshape(-1, 16, 16, 384)
z = z.permute(0, 3, 1, 2)
torch.save(z.cpu(), "dinov3_sp_feature.pt")



# ==== 定义 Decoder ====
decoder = Decoder(**ddconfig).to(device)
# decoder_state_dict = torch.load("/ytech_m2v3_hdd/yuanziyang/sml/DiffMoE_research_local/LightningDiT/last.ckpt")["state_dict"]
decoder_state_dict = torch.load("/ytech_m2v3_hdd/yuanziyang/sml/FVG/model_vitsp16.pt")
print(decoder_state_dict.keys())
decoder_rename_dict = {}
for k, v in decoder_state_dict.items():
    if "decoder" in k:
        # if "conv_in.weight" in k:
            # continue
        decoder_rename_dict[k[8:]] = v
# print(decoder_rename_dict)


decoder.load_state_dict(decoder_rename_dict, strict=False)


# ==== 随机输入测试 ====
batch_size, h, w = 1, 16, 16
# z = torch.randn(batch_size, ddconfig["z_channels"], h, w)
# z = 0.7 * z + 0.3 * torch.randn(batch_size, ddconfig["z_channels"], h, w).to(device)
z = 1.0 * z + 0.0 * torch.randn(batch_size, ddconfig["z_channels"], h, w).to(device)
out = decoder(z)
print(f"Input latent shape: {z.shape}")
print(f"Decoder output shape: {out.shape}")

import torchvision.utils as vutils

# 先把 [-1,1] 转到 [0,1]
out_vis = (out.clamp(-1, 1) + 1) / 2.0  
# out_vis = out

vutils.save_image(out_vis, "decoder_output.png")
print("保存成功：decoder_output.png")
