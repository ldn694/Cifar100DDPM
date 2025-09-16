import torch
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from settings import NUM_CLASSES
from utils import show_images
import os

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = UNet2DModel.from_pretrained("ddpm-cifar100-32px").to(device)
noise_scheduler = DDPMScheduler.from_pretrained("ddpm-cifar100-32px")
image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)

output_folder = "pipeline_output"
os.makedirs(output_folder, exist_ok=True)

total_images = 2048
batch_size = 256
for i in range(total_images // batch_size):
    with torch.no_grad():
        output = image_pipe(batch_size).images  # list of PIL images
    for j, im in enumerate(output):
        im.save(f"{output_folder}/image_{i * batch_size + j:04d}.png")