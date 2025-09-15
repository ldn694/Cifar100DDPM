import torch
from diffusers import DDPMPipeline
from utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the butterfly pipeline
butterfly_pipeline = DDPMPipeline.from_pretrained(
    "johnowhitaker/ddpm-butterflies-32px"
).to(device)

# Create 8 images
images = butterfly_pipeline(batch_size=8).images

# View the result
make_grid(images).show()