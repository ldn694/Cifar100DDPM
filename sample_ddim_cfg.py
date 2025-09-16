import torch
from diffusers import UNet2DConditionModel, DDIMScheduler, DDPMPipeline
from model.pipeline import ClassConditionalDDIMPipeline
from settings import NUM_CLASSES
from utils import show_images
import os

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = UNet2DConditionModel.from_pretrained("ddim_cfg-cifar100-32px").to(device)
noise_scheduler = DDIMScheduler.from_pretrained("ddim_cfg-cifar100-32px")
image_pipe = ClassConditionalDDIMPipeline(unet=model, scheduler=noise_scheduler)

output_folder = "pipeline_output"
os.makedirs(output_folder, exist_ok=True)

total_images = 2048
batch_size = 256
for i in range(total_images // batch_size):
    with torch.no_grad():
        output = image_pipe(batch_size=batch_size, num_inference_steps=50, cfg_scale=7.5, eta=0.0).images
    for j, im in enumerate(output):
        im.save(f"{output_folder}/image_{i * batch_size + j:04d}.png")


output_folder_fixed_class = "pipeline_output_fixed_class"
os.makedirs(output_folder_fixed_class, exist_ok=True)

total_images = 2000
for i in range(NUM_CLASSES):
    class_labels = torch.full((total_images // NUM_CLASSES,), i, device=device)
    with torch.no_grad():
        output = image_pipe(batch_size=total_images // NUM_CLASSES, class_labels=class_labels, num_inference_steps=50, cfg_scale=7.5, eta=0.0).images
    for j, im in enumerate(output):
        im.save(f"{output_folder_fixed_class}/image_{i:02d}_{j:04d}.png")