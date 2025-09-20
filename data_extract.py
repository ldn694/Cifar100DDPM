import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
import os

torch.manual_seed(0)

image_size = 32
batch_size = 1

preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

train_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=True, transform=preprocess, download=True
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,          # you can use >0 safely now
    pin_memory=True
)

output_folder = "cifar100_train"
os.makedirs(output_folder, exist_ok=True)

# Extract 2000 images and save them to disk
total_images = 2000
image_count = 0
image_count_per_class = 20
image_counts = [0] * 100  # CIFAR-100 has 100 classes
for i, (images, labels) in enumerate(train_dataloader):
    for j in range(images.size(0)):
        if image_count >= total_images:
            break
        if image_counts[labels[j].item()] >= image_count_per_class:
            continue
        img = images[j]
        img = (img * 0.5 + 0.5).clamp(0, 1)  # Unnormalize
        image_id = image_counts[labels[j].item()]
        label_id = labels[j].item()
        torchvision.utils.save_image(img, f"{output_folder}/image_{label_id:02d}_{image_id:04d}.png")
        image_counts[labels[j].item()] += 1
        image_count += 1
    if image_count >= total_images:
        break
