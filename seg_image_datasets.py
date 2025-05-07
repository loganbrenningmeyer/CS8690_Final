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
import os
import json
from pathlib import Path
from tqdm import tqdm

print("Loading images...")

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
    weight=1.0
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

print("Running diffusion...")

generator = torch.Generator(device="cuda").manual_seed(42)   # reproducible


# ----------
# Diffuse realistic image
# ----------
params = {
    'prompt': "asphalt micro-texture, concrete rooftops with slight staining, photogrammetric sharpness, realistic aerial perspective",
    'negative_prompt': "warped perspective,repeated tiling patterns,checkerboard texture, pixelated aliasing",
    'strength': 0.25,
    'num_inference_steps': 100,
    'guidance_scale': 6,
    'control_guidance_start': 0.0,
    'control_guidance_end': 1.0,
    'controlnet_conditioning_scale': 0.8,
    'ip_adapter_conditioning_scale': 2.0,
}

def run_on_dataset(dataset_path,output_path:Path,ref_image):
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name

    output_path = output_path / dataset_name

    os.makedirs(output_path,exist_ok=True)


    opt_dir = dataset_path / "opt"

    tif_files = list(opt_dir.glob("*.tif"))

    for tif_file in tqdm(tif_files,desc=f"{dataset_name}",unit='file'):
        seg_image = dataset_path / "gt_nDSM" / tif_file.name
        seg_image = Image.open(seg_image)
        input_file = Image.open(tif_file)

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
            image                         = input_file,
            control_image                 = seg_image, 
            ip_adapter_image              = ref_image,     #color reference
            generator                     = generator
        ).images[0]

        image_output_path = output_path / tif_file.name
        out.save(image_output_path,format='TIFF')



real_path = "/mnt/data/references/JAX_427_009_013_RIGHT_RGB.tif"
ref_img = Image.open(real_path)
synthetic_datasets = [
    "/mnt/data/terrain_g05_mid_v1",
    "/mnt/data/grid_g05_mid_v2",
    "/mnt/data/terrain_g05_low_v1",
    "/mnt/data/terrain_g05_high_v1",
    "/mnt/data/terrain_g005_mid_v1",
    "/mnt/data/terrain_g005_low_v1",
    "/mnt/data/grid_g005_mid_v2",
    "/mnt/data/terrain_g005_high_v1",
    "/mnt/data/terrain_g1_mid_v1",
    "/mnt/data/terrain_g1_low_v1",
    "/mnt/data/terrain_g1_high_v1",
    "/mnt/data/grid_g005_mid_v1",
    "/mnt/data/grid_g005_low_v1",
    "/mnt/data/grid_g005_high_v1",
    "/mnt/data/grid_g05_mid_v1",
    "/mnt/data/grid_g05_low_v1",
    "/mnt/data/grid_g05_high_v1"
]
output_path = Path('/mnt/controlNet-output/')
for path in synthetic_datasets:
    run_on_dataset(path,output_path,ref_img)