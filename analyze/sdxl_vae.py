import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from diffusers.models import AutoencoderKL

device = "cuda"
dtype = torch.bfloat16
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
vae = vae.to(device, dtype)

# pil2tensor = transforms.Compose(
    # [
        # transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    # ]
# )

pil2tensor = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),  # [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

image = Image.open("/ytech_m2v3_hdd/yuanziyang/sml/Wan2.1/examples/flf2v_input_last_frame.png")
pixel_values = pil2tensor(image).unsqueeze(0).to(device=device, dtype=dtype)

# encode to latents
latents = vae.encode(pixel_values).latent_dist.sample()
torch.save(latents.cpu(), "sdxl_vae.pt")
def tensor_to_pil(tensor):
    image = tensor.detach().cpu().to(torch.float32)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.mul(255).round().to(dtype=torch.uint8)
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image, mode="RGB")

# 不同噪声强度
noise_scales = [0.0]

for scale in noise_scales:
    noisy_latents = latents + scale * torch.randn_like(latents)

    with torch.no_grad():
        sampled_images = vae.decode(noisy_latents).sample
    
    rec_image = tensor_to_pil(sampled_images[0])
    rec_image.save(f"output_sdxl_{scale:.1f}.jpg")

print("保存完成 ✅")
