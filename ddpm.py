import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from diffusers import DDPMScheduler, UNet2DModel, DDPMPipeline
from settings import image_size, batch_size
import tqdm
import matplotlib.pyplot as plt
from utils import show_images
import os

def main():
    output_path = "samples"
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


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
        num_workers=1,          # you can use >0 safely now
        pin_memory=True,
        persistent_workers=True # optional but nice with >0 workers
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    model = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=("DownBlock2D","DownBlock2D","AttnDownBlock2D","AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D"),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    losses = []

    for epoch in range(20):
        tqdm_iter = tqdm.tqdm(train_dataloader, desc=f"Training epoch {epoch}")
        for step, batch in enumerate(tqdm_iter):
            clean_images, _ = batch                   # <- tuple, not dict
            clean_images = clean_images.to(device)

            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            tqdm_iter.set_postfix(loss=f"{loss.item():.4f}")

        if True:
            num_batches = len(train_dataloader)
            loss_last_epoch = sum(losses[-num_batches:]) / num_batches
            print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")

            import numpy as np
            avg_losses = [np.mean(losses[i*num_batches:(i+1)*num_batches])
                          for i in range(epoch+1)]
            plt.figure(figsize=(8,5))
            plt.plot(range(1, len(avg_losses)+1), avg_losses, marker='o')
            plt.xlabel('Epoch'); plt.ylabel('Average Loss'); plt.title('Training Loss per Epoch')
            plt.grid(True); plt.tight_layout()
            plt.savefig(f'loss.png')
            # plt.show()

            # Random starting point (8 random images):
            sample = torch.randn(8, 3, 32, 32).to(device)

            for i, t in enumerate(noise_scheduler.timesteps):

                # Get model pred
                with torch.no_grad():
                    residual = model(sample, t).sample

                # Update sample with step
                sample = noise_scheduler.step(residual, t, sample).prev_sample
            pil_images = show_images(sample)
            # Saved to "samples/epoch_{epoch+1}.png"
            pil_images.save(os.path.join(output_path, f'epoch_{epoch+1}.png'))

    model.save_pretrained("ddpm-cifar100-32px")
    noise_scheduler.save_pretrained("ddpm-cifar100-32px")

if __name__ == "__main__":
    # Optional: make spawn explicit
    # torch.multiprocessing.set_start_method("spawn", force=True)
    main()
