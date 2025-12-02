#!/usr/bin/env python3
"""
Stable Diffusion LoRA Training - Clean Version using Diffusers built-in LoRA
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
# DATASET - NO CHANGES
# ============================================================================


class StableDiffusionDataset(Dataset):
    """Dataset untuk fine-tuning Stable Diffusion"""

    def __init__(self, data_path, tokenizer, resolution=64, is_test=False):
        """
        Args:
            data_path: Path to dataset directory
            tokenizer: CLIP tokenizer
            resolution: Image resolution
            is_test: If True, treat as test dataset where orig folder is optional
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.resolution = resolution
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

        # Resize
        original = original.resize((self.resolution, self.resolution), Image.LANCZOS)
        erased = erased.resize((self.resolution, self.resolution), Image.LANCZOS)
        mask = mask.resize((self.resolution, self.resolution), Image.LANCZOS)

        # Convert to tensor & normalize to [-1, 1]
        original = torch.from_numpy(np.array(original)).float() / 127.5 - 1.0
        erased = torch.from_numpy(np.array(erased)).float() / 127.5 - 1.0
        mask = torch.from_numpy(np.array(mask)).float() / 255.0

        # Permute to CHW
        original = original.permute(2, 0, 1)
        erased = erased.permute(2, 0, 1)
        mask = mask.unsqueeze(0)

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
# SETUP MODELS - USING DIFFUSERS BUILT-IN LoRA
# ============================================================================


def setup_models_diffusers_lora(args):
    """Setup using diffusers' built-in LoRA support"""
    print("Setting up models with diffusers built-in LoRA...")

    # Create pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    # Freeze all components
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    # Enable LoRA for UNet
    print(f"Enabling LoRA with rank={args.lora_rank}...")
    pipe.unet.load_attn_procs(
        args.pretrained_lora_path
        if hasattr(args, "pretrained_lora_path") and args.pretrained_lora_path
        else None,
        adapter_name="default",
    )

    # If no pretrained LoRA, we need to inject it
    if not hasattr(args, "pretrained_lora_path") or not args.pretrained_lora_path:
        from diffusers.models.attention_processor import (
            LoRAAttnProcessor,
            LoRAAttnProcessor2_0,
        )

        # Set attention processor to LoRA
        lora_attn_procs = {}
        for name in pipe.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else pipe.unet.config.cross_attention_dim
            )

            # Get hidden size based on block
            if name.startswith("mid_block"):
                hidden_size = pipe.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipe.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipe.unet.config.block_out_channels[block_id]

            # Use appropriate processor
            if hasattr(F, "scaled_dot_product_attention"):
                attn_processor_class = LoRAAttnProcessor2_0
            else:
                attn_processor_class = LoRAAttnProcessor

            lora_attn_procs[name] = attn_processor_class(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=args.lora_rank,
            )

        pipe.unet.set_attn_processor(lora_attn_procs)

    # Move to device
    pipe = pipe.to(args.device)

    # Extract components
    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # Get trainable parameters (only LoRA)
    trainable_params = []
    for name, param in unet.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    print(f"Trainable parameters: {len(trainable_params)}")
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_name, subfolder="scheduler"
    )

    return tokenizer, text_encoder, vae, unet, noise_scheduler, trainable_params


# ============================================================================
# SIMPLE TRAINING - NO COMPLEXITY
# ============================================================================


def train_simple(args):
    """Simple training without complex LoRA logic"""
    print("Starting simple training...")

    # Setup models
    tokenizer, text_encoder, vae, unet, noise_scheduler, trainable_params = (
        setup_models_diffusers_lora(args)
    )

    # Training dataset
    train_dataset = StableDiffusionDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        resolution=args.resolution,
        is_test=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

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

            # Encode images to latent space
            with torch.no_grad():
                # Convert to right dtype
                if args.device == "cuda":
                    original = original.half()

                latents_original = vae.encode(original).latent_dist.sample() * 0.18215

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

            # Predict noise - SIMPLE CALL
            noise_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
            ).sample

            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"lora_epoch_{epoch + 1}.safetensors"
            )

            # Save LoRA weights
            unet.save_attn_procs(
                args.checkpoint_dir, weight_name=f"lora_epoch_{epoch + 1}.safetensors"
            )
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training completed!")

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "lora_final.safetensors")
    unet.save_attn_procs(args.checkpoint_dir, weight_name="lora_final.safetensors")
    print(f"Final LoRA weights saved to {final_path}")

    return unet


# ============================================================================
# ALTERNATIVE: USE PEFT LIBRARY (MOST RELIABLE)
# ============================================================================


def setup_peft_lora(args):
    """Setup using PEFT library - most reliable"""
    try:
        from peft import LoraConfig, get_peft_model

        print("Setting up PEFT LoRA...")

        # Load models
        tokenizer = CLIPTokenizer.from_pretrained(
            args.model_name, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.model_name, subfolder="text_encoder"
        )
        vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(args.model_name, subfolder="unet")

        # Move to device
        text_encoder = text_encoder.to(args.device)
        vae = vae.to(args.device)
        unet = unet.to(args.device)

        # Freeze models
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)

        # Configure LoRA
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )

        # Convert UNet to PEFT model
        unet = get_peft_model(unet, lora_config)

        # Print trainable parameters
        unet.print_trainable_parameters()

        # Get trainable parameters
        trainable_params = []
        for name, param in unet.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)

        # Noise scheduler
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.model_name, subfolder="scheduler"
        )

        return tokenizer, text_encoder, vae, unet, noise_scheduler, trainable_params

    except ImportError:
        print("PEFT library not found. Please install: pip install peft")
        return None


def train_with_peft(args):
    """Train with PEFT library"""
    print("Training with PEFT LoRA...")

    # Setup with PEFT
    result = setup_peft_lora(args)
    if result is None:
        print("Falling back to simple training...")
        return train_simple(args)

    tokenizer, text_encoder, vae, unet, noise_scheduler, trainable_params = result

    # Training dataset
    train_dataset = StableDiffusionDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        resolution=args.resolution,
        is_test=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

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

            # Encode images to latent space
            with torch.no_grad():
                latents_original = vae.encode(original).latent_dist.sample() * 0.18215

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

            # Predict noise
            noise_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
            ).sample

            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"lora_epoch_{epoch + 1}.pt"
            )

            # Save state dict
            torch.save(unet.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training completed!")

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "lora_final.pt")
    torch.save(unet.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

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
        "--use_peft",
        action="store_true",
        help="Use PEFT library for LoRA (recommended)",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--resolution", type=int, default=64, help="Image resolution")

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
            print("Using CUDA")
        else:
            args.device = "cpu"
            print("Using CPU")

    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Choose training method
    if args.use_peft:
        print("\nUsing PEFT LoRA training...")
        train_with_peft(args)
    else:
        print("\nUsing diffusers built-in LoRA training...")
        train_simple(args)


if __name__ == "__main__":
    main()
