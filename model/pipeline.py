from dataclasses import dataclass
from typing import List, Optional
import torch
from PIL import Image
from diffusers import DDIMScheduler, DiffusionPipeline
from settings import NULL_CLASS, NUM_CLASSES

@dataclass
class ImagePipelineOutput:
    images: List[Image.Image]

class ClassConditionalDDIMPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler: DDIMScheduler, image_size: int = 32):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.image_size = image_size

    @torch.no_grad()
    def __call__(
        self,
        class_labels: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        num_inference_steps: int = 50,
        cfg_scale: float = 3.0,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        device: Optional[torch.device] = None,
    ) -> ImagePipelineOutput:

        device = device or next(self.unet.parameters()).device

        # Resolve batch size / labels
        if class_labels is None:
            if batch_size is None:
                raise ValueError("Provide either class_labels or batch_size.")
            class_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
        else:
            class_labels = class_labels.to(device)
            if batch_size is None:
                batch_size = class_labels.shape[0]

        null_labels = torch.full_like(class_labels, NULL_CLASS, device=device)

        # Set DDIM steps and initialize latent
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        sample = torch.randn(
            batch_size, 3, self.image_size, self.image_size,
            device=device, generator=generator
        )

        for t in self.scheduler.timesteps:
            # Two forward passes (cond / uncond). You can concatenate to speed up.
            eps_cond = self.unet(
                sample, t, class_labels=class_labels, encoder_hidden_states=None
            ).sample
            eps_uncond = self.unet(
                sample, t, class_labels=null_labels, encoder_hidden_states=None
            ).sample

            # CFG combine
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            # DDIM update
            sample = self.scheduler.step(eps, t, sample, eta=eta).prev_sample

        # Post-process to images in [0,255] uint8
        images = self._to_pil(sample) if output_type == "pil" else sample
        return ImagePipelineOutput(images=images)

    def _to_pil(self, sample: torch.Tensor) -> List[Image.Image]:
        """
        sample: (B, 3, H, W) in [-1, 1]
        returns: list of PIL.Image
        """
        sample = (sample.clamp(-1, 1) + 1) / 2.0           # [0,1]
        sample = (sample * 255.0).round().to(torch.uint8)  # [0,255] uint8
        sample = sample.detach().cpu().permute(0, 2, 3, 1).numpy()  # (B,H,W,C)
        return [Image.fromarray(img) for img in sample]
