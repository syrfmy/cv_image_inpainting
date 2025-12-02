#!/usr/bin/env python3
"""
Stable Diffusion LoRA Training - Corrected for UNet dimensions
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Import from diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# ============================================================================
# DATASET WITH PROPER RESIZING
# ============================================================================


class StableDiffusionDataset(Dataset):
    """Dataset untuk fine-tuning Stable Diffusion"""

    def __init__(self, data_path, tokenizer, image_size=512, is_test=False):
        """
        Args:
            data_path: Path to dataset directory
            tokenizer: CLIP tokenizer
            image_size: Input image size (512 for SD 1.5)
            is_test: If True, treat as test dataset where orig folder is optional
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.image_size = image_size  # SD 1.5 expects 512x512 images
        self.is_test = is_test

        self.erased_dir = self.data_path / "erased"
        self.masks_dir = self.data_path / "masks"
        self.orig_dir = self.data_path / "orig"

        # Validasi folder required
        assert self.erased_dir.exists(), f"Folder {self.erased_dir} tidak ditemukan"
        assert self.masks_dir.exists(), f"Folder {self.masks_dir} tidak ditemukan"

        # Orig folder hanya required untuk training
        if not self.is_test:
            assert self.orig_dir.exists(), (
                f"Folder {self.orig_dir} tidak ditemukan untuk training dataset"
            )

        # Cari semua file erased
        self.image_files = []
        for erased_path in sorted(self.erased_dir.glob("*_erased.png")):
            filename = erased_path.stem
            base_name = filename.replace("_erased", "")

            # Extract emoji_id (first part before __)
            emoji_id = base_name.split("__")[0]

            orig_filename = f"{base_name}.png"
            orig_path = self.orig_dir / orig_filename

            # Construct mask path: {base_name}_mask.png
            mask_filename = f"{base_name}_mask.png"
            mask_path = self.masks_dir / mask_filename

            # Validasi mask selalu harus ada
            if not mask_path.exists():
                print(f"Warning: Mask file {mask_path} tidak ditemukan")
                continue

            # Untuk test dataset, orig boleh tidak ada
            if self.is_test:
                if orig_path.exists():
                    self.image_files.append(
                        {
                            "original": orig_path,
                            "erased": erased_path,
                            "mask": mask_path,
                            "emoji_id": emoji_id,
                            "filename": base_name,
                            "has_original": True,
                        }
                    )
                else:
                    # Untuk test tanpa original, kita masih bisa gunakan erased sebagai placeholder
                    self.image_files.append(
                        {
                            "original": None,
                            "erased": erased_path,
                            "mask": mask_path,
                            "emoji_id": emoji_id,
                            "filename": base_name,
                            "has_original": False,
                        }
                    )
            else:
                # Untuk training, orig harus ada
                if not orig_path.exists():
                    print(
                        f"Warning: Original file {orig_path} tidak ditemukan untuk training dataset"
                    )
                    continue

                self.image_files.append(
                    {
                        "original": orig_path,
                        "erased": erased_path,
                        "mask": mask_path,
                        "emoji_id": emoji_id,
                        "filename": base_name,
                        "has_original": True,
                    }
                )

        print(f"Found {len(self.image_files)} samples in {data_path}")
        if len(self.image_files) > 0:
            sample = self.image_files[0]
            print(
                f"Example: emoji_id={sample['emoji_id']}, filename={sample['filename']}"
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        item = self.image_files[idx]

        # Load images
        erased = Image.open(item["erased"]).convert("RGB")
        mask = Image.open(item["mask"]).convert("L")

        # Load original or use erased as placeholder
        if item["has_original"]:
            original = Image.open(item["original"]).convert("RGB")
        else:
            original = erased.copy()

        # Resize to 512x512 for Stable Diffusion
        original = original.resize((self.image_size, self.image_size), Image.LANCZOS)
        erased = erased.resize((self.image_size, self.image_size), Image.LANCZOS)
        mask = mask.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Convert to tensor & normalize to [-1, 1]
        original = torch.from_numpy(np.array(original)).float() / 127.5 - 1.0
        erased = torch.from_numpy(np.array(erased)).float() / 127.5 - 1.0
        mask = torch.from_numpy(np.array(mask)).float() / 255.0

        # Permute to CHW
        original = original.permute(2, 0, 1)
        erased = erased.permute(2, 0, 1)
        mask = mask.unsqueeze(0)  # Add channel dimension

        # Simple prompt with only emoji_id
        emoji_id = item["emoji_id"]
        prompt = f"emoji {emoji_id}"

        # Tokenize prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "original": original,
            "erased": erased,
            "mask": mask,
            "input_ids": text_inputs.input_ids[0],
            "emoji_id": emoji_id,
            "has_original": item["has_original"],
            "filename": item["filename"],
        }


# ============================================================================
# SETUP MODELS WITH CORRECT DIMENSIONS
# ============================================================================


def setup_models_correct(args):
    """Setup models with correct dimensions for Stable Diffusion"""
    print("Setting up models with correct dimensions...")

    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer")

    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_name, subfolder="text_encoder"
    )
    text_encoder.to(args.device)
    text_encoder.requires_grad_(False)

    # Load VAE
    vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae")
    vae.to(args.device)
    vae.requires_grad_(False)

    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(args.model_name, subfolder="unet")
    unet.to(args.device)

    # Freeze UNet initially
    unet.requires_grad_(False)

    # Enable LoRA using diffusers built-in method
    print(f"Setting up LoRA with rank={args.lora_rank}...")

    from diffusers.models.attention_processor import (
        LoRAAttnProcessor,
        LoRAAttnProcessor2_0,
    )

    # Set attention processors to LoRA
    lora_attn_procs = {}

    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )

        # Get hidden size based on block
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            hidden_size = None

        if hidden_size is None:
            continue

        # Choose processor class
        if hasattr(F, "scaled_dot_product_attention"):
            attn_processor_class = LoRAAttnProcessor2_0
        else:
            attn_processor_class = LoRAAttnProcessor

        lora_attn_procs[name] = attn_processor_class(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.lora_rank,
        )

    # Set the processors
    unet.set_attn_processor(lora_attn_procs)

    # Collect trainable parameters (only LoRA)
    trainable_params = []
    for name, param in unet.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            trainable_params.append(param)

    print(f"Trainable LoRA parameters: {sum(p.numel() for p in trainable_params):,}")

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_name, subfolder="scheduler"
    )

    return tokenizer, text_encoder, vae, unet, noise_scheduler, trainable_params


# ============================================================================
# TRAINING FUNCTION WITH CORRECT LATENT DIMENSIONS
# ============================================================================


def train_correct(args):
    """Training with correct latent dimensions"""
    print("Starting training with correct dimensions...")

    # Setup models
    tokenizer, text_encoder, vae, unet, noise_scheduler, trainable_params = (
        setup_models_correct(args)
    )

    # Training dataset
    train_dataset = StableDiffusionDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        image_size=512,  # Stable Diffusion expects 512x512
        is_test=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    print(f"Training on {len(train_dataset)} samples")
    print(f"Image size: 512x512 (SD 1.5 standard)")
    print(f"Latent size: 64x64 (512/8)")

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    unet.train()

    for epoch in range(args.epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            original = batch["original"].to(args.device)
            input_ids = batch["input_ids"].to(args.device)

            # Encode images to latent space (512 -> 64 latents)
            with torch.no_grad():
                # Encode to latents: (B, 3, 512, 512) -> (B, 4, 64, 64)
                latents_original = vae.encode(original).latent_dist.sample()

                # Scale latents (following standard SD practice)
                latents_original = latents_original * 0.18215

                # Get text embeddings
                if args.use_text_conditioning:
                    encoder_hidden_states = text_encoder(input_ids)[0]
                else:
                    encoder_hidden_states = None

            # Sample noise
            noise = torch.randn_like(latents_original)

            # Sample timestep
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents_original.shape[0],),
                device=args.device,
            ).long()

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(
                latents_original, noise, timesteps
            )

            # Predict noise - LATENTS SHOULD BE (B, 4, 64, 64)
            noise_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
            ).sample

            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()

            # Print debug info occasionally
            if batch_idx == 0:
                print(f"\nDebug info:")
                print(
                    f"  Original shape: {original.shape}"
                )  # Should be (B, 3, 512, 512)
                print(
                    f"  Latents shape: {latents_original.shape}"
                )  # Should be (B, 4, 64, 64)
                print(
                    f"  Noise pred shape: {noise_pred.shape}"
                )  # Should be (B, 4, 64, 64)
                print(
                    f"  Encoder hidden states: {encoder_hidden_states.shape if encoder_hidden_states is not None else 'None'}"
                )

            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"lora_epoch_{epoch + 1}.safetensors"
            )

            # Save LoRA weights using diffusers method
            unet.save_attn_procs(
                args.checkpoint_dir, weight_name=f"lora_epoch_{epoch + 1}.safetensors"
            )
            print(f"Checkpoint saved to {checkpoint_path}")

            # Also save optimizer state
            optimizer_path = os.path.join(
                args.checkpoint_dir, f"optimizer_epoch_{epoch + 1}.pt"
            )
            torch.save(optimizer.state_dict(), optimizer_path)

    print("Training completed!")

    # Save final LoRA weights
    final_path = os.path.join(args.checkpoint_dir, "lora_final.safetensors")
    unet.save_attn_procs(args.checkpoint_dir, weight_name="lora_final.safetensors")
    print(f"Final LoRA weights saved to {final_path}")

    return unet


# ============================================================================
# ALTERNATIVE: FINE-TUNE ON 64x64 DIRECTLY (CUSTOM DIMENSIONS)
# ============================================================================


def setup_models_64x64(args):
    """Setup for 64x64 images - custom UNet configuration"""
    print("Setting up for 64x64 images...")

    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer")

    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_name, subfolder="text_encoder"
    )
    text_encoder.to(args.device)
    text_encoder.requires_grad_(False)

    # Load VAE
    vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae")
    vae.to(args.device)
    vae.requires_grad_(False)

    # Load UNet - but we need to handle 64x64 images
    # For 64x64 images, latents will be 8x8 (64/8)
    unet = UNet2DConditionModel.from_pretrained(args.model_name, subfolder="unet")

    # Check UNet configuration
    print(f"UNet config:")
    print(f"  Sample size: {unet.config.sample_size}")  # Should be 64 for SD 1.5
    print(f"  In channels: {unet.config.in_channels}")  # Should be 4
    print(f"  Out channels: {unet.config.out_channels}")  # Should be 4

    # For 64x64 images, we need to handle scaling
    # VAE downsampling factor is 8, so 64x64 -> 8x8 latents
    unet.to(args.device)

    # Enable LoRA
    print(f"Setting up LoRA for 64x64 training...")

    from diffusers.models.attention_processor import (
        LoRAAttnProcessor,
        LoRAAttnProcessor2_0,
    )

    lora_attn_procs = {}

    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )

        # Get hidden size
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            hidden_size = None

        if hidden_size is None:
            continue

        # Choose processor
        if hasattr(F, "scaled_dot_product_attention"):
            attn_processor_class = LoRAAttnProcessor2_0
        else:
            attn_processor_class = LoRAAttnProcessor

        lora_attn_procs[name] = attn_processor_class(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.lora_rank,
        )

    unet.set_attn_processor(lora_attn_procs)

    # Freeze UNet, only train LoRA
    unet.requires_grad_(False)

    # Collect trainable parameters
    trainable_params = []
    for name, param in unet.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            trainable_params.append(param)

    print(f"Trainable LoRA parameters: {sum(p.numel() for p in trainable_params):,}")

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_name, subfolder="scheduler"
    )

    return tokenizer, text_encoder, vae, unet, noise_scheduler, trainable_params


def train_64x64(args):
    """Train on 64x64 images directly"""
    print("Training on 64x64 images...")

    # Setup models
    tokenizer, text_encoder, vae, unet, noise_scheduler, trainable_params = (
        setup_models_64x64(args)
    )

    # Training dataset - use 64x64 images
    train_dataset = StableDiffusionDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        image_size=64,  # Use 64x64 images
        is_test=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    print(f"Training on {len(train_dataset)} 64x64 samples")
    print(f"Latent size will be: 8x8 (64/8)")

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    unet.train()

    for epoch in range(args.epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            original = batch["original"].to(args.device)
            input_ids = batch["input_ids"].to(args.device)

            # Encode 64x64 images to latent space (64 -> 8 latents)
            with torch.no_grad():
                # For 64x64 images, VAE will output 8x8 latents
                latents_original = vae.encode(original).latent_dist.sample()
                latents_original = latents_original * 0.18215

                # Get text embeddings
                if args.use_text_conditioning:
                    encoder_hidden_states = text_encoder(input_ids)[0]
                else:
                    encoder_hidden_states = None

            # Debug shapes
            if batch_idx == 0 and epoch == 0:
                print(f"\nDebug shapes for 64x64 training:")
                print(f"  Input images: {original.shape}")  # (B, 3, 64, 64)
                print(f"  Latents: {latents_original.shape}")  # Should be (B, 4, 8, 8)
                print(f"  UNet sample size config: {unet.config.sample_size}")

            # Sample noise
            noise = torch.randn_like(latents_original)

            # Sample timestep
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents_original.shape[0],),
                device=args.device,
            ).long()

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(
                latents_original, noise, timesteps
            )

            # Predict noise
            noise_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
            ).sample

            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"lora_64x64_epoch_{epoch + 1}.safetensors"
            )
            unet.save_attn_procs(
                args.checkpoint_dir,
                weight_name=f"lora_64x64_epoch_{epoch + 1}.safetensors",
            )
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training completed!")

    # Save final
    final_path = os.path.join(args.checkpoint_dir, "lora_64x64_final.safetensors")
    unet.save_attn_procs(
        args.checkpoint_dir, weight_name="lora_64x64_final.safetensors"
    )
    print(f"Final LoRA weights saved to {final_path}")

    return unet


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train Stable Diffusion with LoRA")

    # Dataset paths
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training dataset"
    )

    # Model settings
    parser.add_argument(
        "--model_name",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base model name",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=4, help="Rank for LoRA (4-16 typically)"
    )
    parser.add_argument(
        "--use_text_conditioning", action="store_true", help="Use text conditioning"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        choices=[64, 512],
        default=64,
        help="Image size for training (64 or 512)",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )

    # Checkpoint settings
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )

    # System settings
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of data loader workers"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Disable CUDA even if available"
    )

    args = parser.parse_args()

    # Device setup
    if args.device is None:
        if not args.no_cuda and torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    print(f"Using device: {args.device}")
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Choose training method based on image size
    if args.image_size == 512:
        print("\nTraining on 512x512 images (standard SD 1.5)...")
        train_correct(args)
    else:  # 64x64
        print("\nTraining on 64x64 images...")
        train_64x64(args)


if __name__ == "__main__":
    main()
