#!/usr/bin/env python3
"""
Optimized LoRA Training - Fixed Version
"""

import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler

# PEFT imports
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

logger = get_logger(__name__)


def prepare_inpainting_inputs(erased_image, mask_image):
    """Prepare inputs for inpainting training"""
    # Convert erased image to tensor [-1, 1]
    erased_np = np.array(erased_image.convert("RGB"))
    erased_np = erased_np[None].transpose(0, 3, 1, 2)
    erased_tensor = torch.from_numpy(erased_np).to(dtype=torch.float32) / 127.5 - 1.0

    # Convert mask: white=damage (1), black=intact (0)
    mask_np = np.array(mask_image.convert("L"))
    mask_np = mask_np.astype(np.float32) / 255.0
    mask_np = mask_np[None, None]  # Add batch and channel dims
    mask_tensor = torch.from_numpy(mask_np)

    return mask_tensor, erased_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimized LoRA Training for Inpainting"
    )

    # Required arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Path to training dataset (should contain erased/, masks/, orig/ folders).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-4hour-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Training parameters
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,  # Reduced to avoid memory issues
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,  # Reduced for testing
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )

    # LoRA parameters
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,  # Reduced for speed
        help="Rank of LoRA layers.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Alpha parameter for LoRA scaling.",
    )

    # System parameters
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,  # Set to 0 to avoid warnings
        help="Number of worker processes for data loading.",
    )

    args = parser.parse_args()

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


class InpaintingDataset(Dataset):
    """
    Dataset for inpainting training.
    """

    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.data_root = Path(data_root)

        # Load paths
        self.erased_dir = self.data_root / "erased"
        self.masks_dir = self.data_root / "masks"
        self.orig_dir = self.data_root / "orig"

        # Verify folders
        assert self.erased_dir.exists(), f"erased folder not found: {self.erased_dir}"
        assert self.masks_dir.exists(), f"masks folder not found: {self.masks_dir}"
        assert self.orig_dir.exists(), f"orig folder not found: {self.orig_dir}"

        # Get all erased files
        self.erased_files = sorted(list(self.erased_dir.glob("*_erased.png")))
        print(f"Found {len(self.erased_files)} training samples")

        # Simple image transform
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size),
            ]
        )

    def __len__(self):
        return len(self.erased_files)

    def __getitem__(self, index):
        # Get paths
        erased_path = self.erased_files[index]
        filename = erased_path.stem
        base_name = filename.replace("_erased", "")

        mask_path = self.masks_dir / f"{base_name}_mask.png"
        orig_path = self.orig_dir / f"{base_name}.png"

        # Fixed prompt
        prompt = "a damaged picture of a single emoji that needs to be repaired"

        # Load images
        erased_image = Image.open(erased_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")
        orig_image = Image.open(orig_path).convert("RGB")

        # Apply transforms
        erased_image = self.transform(erased_image)
        orig_image = self.transform(orig_image)
        mask_image = mask_image.resize(orig_image.size, Image.NEAREST)

        # Tokenize prompt
        prompt_ids = self.tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return {
            "erased_image": erased_image,
            "orig_image": orig_image,
            "mask": mask_image,
            "prompt_ids": prompt_ids,
        }


def collate_fn(examples, tokenizer):
    """Collate function for the dataloader"""
    # Tokenization
    input_ids = [example["prompt_ids"] for example in examples]
    input_ids = tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_tensors="pt"
    ).input_ids

    # Prepare lists for tensors
    orig_tensors = []
    masks = []
    erased_tensors = []

    for example in examples:
        # Original image (ground truth)
        orig_pil = example["orig_image"]
        orig_np = np.array(orig_pil.convert("RGB"))
        orig_np = orig_np.transpose(2, 0, 1)  # HWC to CHW
        orig_tensor = torch.from_numpy(orig_np).float() / 127.5 - 1.0
        orig_tensors.append(orig_tensor)

        # Prepare mask and erased image
        mask, erased = prepare_inpainting_inputs(
            example["erased_image"], example["mask"]
        )
        masks.append(mask.squeeze(0))  # Remove batch dimension
        erased_tensors.append(erased.squeeze(0))  # Remove batch dimension

    # Stack tensors
    orig_tensors = torch.stack(orig_tensors)
    masks = torch.stack(masks)
    erased_tensors = torch.stack(erased_tensors)

    return {
        "input_ids": input_ids,
        "orig_images": orig_tensors,  # Shape: [B, 3, H, W]
        "masks": masks,  # Shape: [B, 1, H, W]
        "erased_images": erased_tensors,  # Shape: [B, 3, H, W]
    }


def main():
    args = parse_args()

    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and models
    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    # Freeze base models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Set dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to device
    device = accelerator.device
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    # Setup LoRA
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet.add_adapter(lora_config)

    # Set to train mode
    unet.train()

    # Collect trainable parameters
    trainable_params = []
    for name, param in unet.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    total_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total_params:,}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    # Load dataset
    print("Loading dataset...")
    train_dataset = InpaintingDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
    )

    # Create dataloader with custom collate
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer),
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # Setup scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Log info
    print(f"\nStarting training with {len(train_dataset)} samples")
    print(f"Target steps: {args.max_train_steps}")
    print(f"Batch size: {args.train_batch_size}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Learning rate: {args.learning_rate}")

    # Training loop
    global_step = 0
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(args.num_train_epochs):
        unet.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Debug: Print shapes
                if global_step == 0:
                    print(f"\nBatch shapes at step 0:")
                    print(f"  orig_images: {batch['orig_images'].shape}")
                    print(f"  masks: {batch['masks'].shape}")
                    print(f"  erased_images: {batch['erased_images'].shape}")
                    print(f"  input_ids: {batch['input_ids'].shape}")

                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Convert images to latents - FIXED VERSION
                with torch.no_grad():
                    # Target latents (original image)
                    # Ensure correct shape: [B, 3, H, W] -> [B, C, H, W]
                    target_latents = vae.encode(
                        batch["orig_images"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    target_latents = target_latents * vae.config.scaling_factor

                    # Input latents (erased image)
                    erased_latents = vae.encode(
                        batch["erased_images"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    erased_latents = erased_latents * vae.config.scaling_factor

                # Debug: Print latent shapes
                if global_step == 0:
                    print(f"\nLatent shapes at step 0:")
                    print(f"  target_latents: {target_latents.shape}")
                    print(f"  erased_latents: {erased_latents.shape}")

                # Resize mask to latent size
                mask = batch["masks"].to(dtype=weight_dtype)
                mask_resized = torch.nn.functional.interpolate(
                    mask,
                    size=(args.resolution // 8, args.resolution // 8),
                    mode="nearest",
                )

                # Sample noise
                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]

                # Sample timesteps
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=target_latents.device,
                ).long()

                # Add noise to target latents
                noisy_latents = noise_scheduler.add_noise(
                    target_latents, noise, timesteps
                )

                # Debug: Print shapes before concatenation
                if global_step == 0:
                    print(f"\nShapes before concatenation:")
                    print(f"  noisy_latents: {noisy_latents.shape}")
                    print(f"  mask_resized: {mask_resized.shape}")
                    print(f"  erased_latents: {erased_latents.shape}")

                # Concatenate for inpainting: [noisy_latents, mask, erased_latents]
                model_input = torch.cat(
                    [noisy_latents, mask_resized, erased_latents], dim=1
                )

                if global_step == 0:
                    print(f"  model_input shape: {model_input.shape}")

                # Predict noise
                noise_pred = unet(model_input, timesteps, encoder_hidden_states).sample

                # Calculate loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log loss
                if global_step % 10 == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                    }
                    progress_bar.set_postfix(logs)

                # Save checkpoint
                if global_step % 500 == 0 and accelerator.is_main_process:
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)
                    unet.save_attn_procs(os.path.join(save_path, "lora_weights"))
                    print(f"\nSaved checkpoint at step {global_step}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Save final model
    if accelerator.is_main_process:
        print(f"\nSaving final model to {args.output_dir}")
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args.output_dir)

        # Also save as safetensors
        from safetensors.torch import save_file

        peft_state_dict = get_peft_model_state_dict(unet)
        save_file(
            peft_state_dict,
            os.path.join(args.output_dir, "pytorch_lora_weights.safetensors"),
        )

        print(f"Training completed in {global_step} steps")
        print(f"LoRA weights saved to {args.output_dir}")


if __name__ == "__main__":
    main()
