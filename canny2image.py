# canny2image.py

print("Loading libraries...")

import cv2
import torch
from PIL import Image
import numpy as np
from controlnet_aux import HEDdetector  # still needed for placeholder import, but not used below
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
src_img = Image.open(img_path)

real_path = "references/JAX_427_009_013_RIGHT_RGB.tif"
ref_img = Image.open(real_path)

# ----------
# Canny edge map
# ----------
print("Generating Canny edge map...")

# convert to grayscale numpy array
gray = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2GRAY)

# optional blur to reduce noise before edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.5)

# apply Canny: adjust thresholds to taste
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# convert single-channel edges back to 3-channel PIL image
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
canny_img = Image.fromarray(edges_rgb)

# ----------
# Control via Canny Map
# ----------
base_id = "runwayml/stable-diffusion-v1-5"
canny_id = "lllyasviel/sd-controlnet-canny"

control_canny = ControlNetModel.from_pretrained(
    canny_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to("cuda")

# ----------
# Load diffusion pipeline
# ----------
print("Initializing diffusion pipeline & IP-Adapter...")

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    base_id,
    controlnet=control_canny,
    torch_dtype=torch.float16
).to("cuda")

pipe.load_ip_adapter(
    "h94/IP-Adapter",              # repo root  (public)
    "models",                      # sub-folder with the weights
    "ip-adapter-plus_sd15.safetensors",
    weight=1
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# -- Create Torch random Generator for reproducibility
generator = torch.Generator(device="cuda").manual_seed(42)

# ----------
# Diffuse realistic image
# ----------
params = {
    'prompt': 'asphalt micro-texture, concrete rooftops with slight staining, photogrammetric sharpness, realistic aerial perspective',
    'negative_prompt': 'warped perspective,repeated tiling patterns,checkerboard texture, pixelated aliasing',
    'strength': 0.25,
    'num_inference_steps': 100,
    'guidance_scale': 6,
    'control_guidance_start': 0.0,
    'control_guidance_end': 1.0,
    'controlnet_conditioning_scale': 0.8,
    'ip_adapter_conditioning_scale': 2.0,
}

out = pipe(
    prompt=params['prompt'],
    negative_prompt=params['negative_prompt'],
    strength=params['strength'],
    num_inference_steps=params['num_inference_steps'],
    guidance_scale=params['guidance_scale'],
    control_guidance_start=params['control_guidance_start'],
    control_guidance_end=params['control_guidance_end'],
    controlnet_conditioning_scale=params['controlnet_conditioning_scale'],
    ip_adapter_conditioning_scale=params['ip_adapter_conditioning_scale'],
    image=src_img,
    control_image=canny_img,
    ip_adapter_image=ref_img,
    generator=generator
).images[0]

# ----------
# Save outputs & parameters
# ----------
print("Saving realistic image & Canny map & params...")

run_name = 'canny_test'
out_dir = os.path.join('outputs', run_name)
os.makedirs(out_dir, exist_ok=True)

# Save generated image
out.save(os.path.join(out_dir, f'{run_name}.tif'), format='TIFF')

# Save Canny edge map for reference
canny_img.save(os.path.join(out_dir, 'canny.png'))

# Save parameters to JSON
with open(os.path.join(out_dir, 'params.json'), 'w') as f:
    json.dump(params, f, indent=4)