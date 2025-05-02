print("Loading libraries...")

from controlnet_aux import HEDdetector
import cv2, torch
from PIL import Image
import numpy as np
from diffusers import (
    ControlNetModel, 
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)

print("Loading images...")

img_path = "inputs/0000002193-1_0.tif"
src_img   = Image.open(img_path).convert("RGB")

real_path = "references/real_satellite.png"
ref_img = Image.open(real_path).convert("RGB")

seg_path = "seg/0000002193-1_0.tif"
seg_img = Image.open(seg_path).convert("RGB")

# ----------
# Control via Segmentation Map
# ----------
base_id = "runwayml/stable-diffusion-v1-5"
seg_id = "lllyasviel/sd-controlnet-seg"

control_seg = ControlNetModel.from_pretrained(
    seg_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to("cuda")

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    base_id,
    controlnet=control_seg,    # Multi-ControlNet
    torch_dtype=torch.float16
).to("cuda")

print("Initializing diffusion pipeline...")

pipe.load_ip_adapter(
    "h94/IP-Adapter",          # repo root  (public)
    "models",    # sub-folder with the weights
    "ip-adapter-plus_sd15.safetensors",  # file name (or None to auto-pick)
    weight=1               # adapter strength
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

print("Running diffusion...")

generator = torch.Generator(device="cuda").manual_seed(42)   # reproducible

out = pipe(
        prompt             = "high-resolution satellite photograph of Earth, realistic lighting, natural terrain colors, true-to-life vegetation and water tones, sharp detail",
        negative_prompt    = "colorful shadows, neon tint, cartoon texture, oversaturated",
        image              = src_img,
        control_image      = seg_img, 
        ip_adapter_image   = ref_img,     #color reference
        strength           = 0.15,        # denoise — keep geometry!
        num_inference_steps= 75,
        guidance_scale     = 8,           # CFG
        control_guidance_start = 0.0,  # HED on from the first step …
        control_guidance_end   = 0.9,  # … but disabled after 50 % of steps
        controlnet_conditioning_scale=1.0,  # weights we chose earlier
        ip_adapter_conditioning_scale=1.0,
        generator          = generator
).images[0]

print("Saving realistic image...")

out.save("outputs/seg_test_3.png")