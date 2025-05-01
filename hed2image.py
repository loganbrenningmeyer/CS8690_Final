from controlnet_aux import HEDdetector
import cv2, torch
from PIL import Image
import numpy as np
from diffusers import (
    ControlNetModel, 
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)

img_path = "inputs/0000002022-1_1.tif"
src_img   = Image.open(img_path).convert("RGB")

# ----- HED edge map (structure lock) -----
hed = HEDdetector.from_pretrained('lllyasviel/Annotators')
hed_map = hed(src_img)                        # returns a PIL image, 512×512 by default
hed_map = cv2.GaussianBlur(
            cv2.cvtColor(np.array(hed_map), cv2.COLOR_RGB2GRAY),
            (0, 0), sigmaX=3)                 # soft-blur: mid-control only needs hints
hed_map = Image.fromarray(hed_map)

base_id = "runwayml/stable-diffusion-v1-5"
hed_id = "lllyasviel/sd-controlnet-hed"

control_hed = ControlNetModel.from_pretrained(hed_id , torch_dtype=torch.float16)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base_id,
            controlnet=control_hed,    # Multi-ControlNet
            torch_dtype=torch.float16
)

pipe.load_ip_adapter(
    "h94/IP-Adapter",          # repo root  (public)
    "models",    # sub-folder with the weights
    "ip-adapter-plus_sd15.safetensors",  # file name (or None to auto-pick)
    weight=1               # adapter strength
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")



generator = torch.Generator(device="cuda").manual_seed(42)   # reproducible

out = pipe(
        prompt             = "satellite photo",          # or a short 1-2 word tag like "satellite photo"
        negative_prompt    = "colorful shadows, neon tint, cartoon texture, oversaturated",
        image              = src_img,
        control_image      = hed_map, 
        ip_adapter_image   = src_img,     #color reference
        strength           = 0.25,        # denoise — keep geometry!
        num_inference_steps= 32,
        guidance_scale     = 6,           # CFG
        control_guidance_start = 0.0,  # HED on from the first step …
        control_guidance_end   = 0.75,  # … but disabled after 50 % of steps
        controlnet_conditioning_scale=0.35,  # weights we chose earlier
        generator          = generator
).images[0]

print("Saving realistic image...")

out.save("/mnt/project/outputs/satellite_realistic.png")