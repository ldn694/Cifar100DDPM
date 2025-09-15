import torch
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from utils import show_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = UNet2DModel.from_pretrained("ddpm-butterflies-32px").to(device)
noise_scheduler = DDPMScheduler.from_pretrained("ddpm-butterflies-32px")
image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)

# Random starting point (8 random images):
sample = torch.randn(8, 3, 32, 32).to(device)

for i, t in enumerate(noise_scheduler.timesteps):

    # Get model pred
    with torch.no_grad():
        residual = model(sample, t).sample

    # Update sample with step
    sample = noise_scheduler.step(residual, t, sample).prev_sample

show_images(sample).show()