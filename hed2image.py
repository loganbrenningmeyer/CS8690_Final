print("Loading libraries...")

from controlnet_aux import HEDdetector
import torch
from PIL import Image
import numpy as np
from diffusers import (
    ControlNetModel, 
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)
import json
import os

# ----------
# Load source/control/reference images
# ----------
print("Loading images...")

img_path = "inputs/0000002193-1_0.tif"
src_img   = Image.open(img_path)

real_path = "references/JAX_427_009_013_RIGHT_RGB.tif"
ref_img = Image.open(real_path)

seg_path = "seg/0000002193-1_0.tif"
seg_img = Image.open(seg_path)

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

# ----------
# Load diffusion pipeline
# ----------
print("Initializing diffusion pipeline & IP-Adapter...")

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    base_id,
    controlnet=control_seg,    # Multi-ControlNet
    torch_dtype=torch.float16
).to("cuda")

pipe.load_ip_adapter(
    "h94/IP-Adapter",          # repo root  (public)
    "models",    # sub-folder with the weights
    "ip-adapter-plus_sd15.safetensors",  # file name (or None to auto-pick)
    weight=1               # adapter strength
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# -- Create Torch random Generator
generator = torch.Generator(device="cuda").manual_seed(42)   # reproducible

# ----------
# Diffuse realistic image
# ----------
params = {
    'prompt': 'asphalt micro-texture, concrete rooftops with slight staining, photogrammetric sharpness, realistic aerial perspective',
    'negative_prompt': 'warped perspective,repeated tiling patterns,checkerboard texture, pixelated aliasing',
    'strength': 0.1,
    'num_inference_steps': 50,
    'guidance_scale': 8,
    'control_guidance_start': 0.0,
    'control_guidance_end': 1.0,
    'controlnet_conditioning_scale': 0.8,
    'ip_adapter_conditioning_scale': 2.0,
}

out = pipe(
        prompt                        = params['prompt'],
        negative_prompt               = params['negative_prompt'],
        strength                      = params['strength'],        # denoise — keep geometry!
        num_inference_steps           = params['num_inference_steps'],
        guidance_scale                = params['guidance_scale'],           # CFG
        control_guidance_start        = params['control_guidance_start'],  # HED on from the first step …
        control_guidance_end          = params['control_guidance_end'],  # … but disabled after 50 % of steps
        controlnet_conditioning_scale = params['controlnet_conditioning_scale'],  # weights we chose earlier
        ip_adapter_conditioning_scale = params['ip_adapter_conditioning_scale'],
        image                         = src_img,
        control_image                 = seg_img, 
        ip_adapter_image              = ref_img,     #color reference
        generator                     = generator
).images[0]

# ----------
# Save image and parameters to JSON
# ----------
print("Saving realistic image & params...")

run_name = 'seg_test_13'
out_dir = os.path.join('outputs', run_name)

# -- Create output directory
os.makedirs(out_dir, exist_ok=True)

# -- Save image
out.save(os.path.join(out_dir, f'{run_name}.tif'), format='TIFF')

# -- Save params to JSON
with open(os.path.join(out_dir, f'params.json'), 'w') as f:
    json.dump(params, f, indent=4)