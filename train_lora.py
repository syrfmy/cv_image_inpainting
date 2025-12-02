#!/usr/bin/env python3
"""
Stable Diffusion LoRA Training - Fixed Version
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
# DATASET (SAME AS BEFORE)
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
# SIMPLE APPROACH: TRAIN SELECTED LAYERS (NO LoRA COMPLEXITY)
# ============================================================================


def setup_models_simple(args):
    """Simple setup without LoRA - train selected attention layers"""
    print("Loading models...")

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

    # Freeze all parameters first
    unet.requires_grad_(False)

    # Unfreeze only attention layers (q, k, v, proj_out)
    trainable_params = []
    for name, param in unet.named_parameters():
        # Train attention query, key, value, and output projection layers
        if any(x in name for x in ["to_q", "to_k", "to_v", "to_out.0", "proj_out"]):
            param.requires_grad = True
            trainable_params.append(param)
        # Also train attention processors if they exist
        elif "processor" in name and "weight" in name:
            param.requires_grad = True
            trainable_params.append(param)

    print(f"Trainable parameters: {len(trainable_params)}")
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_name, subfolder="scheduler"
    )

    return tokenizer, text_encoder, vae, unet, noise_scheduler, trainable_params


# ============================================================================
# WORKING LoRA USING DIFFUSERS BUILT-IN
# ============================================================================


def setup_models_diffusers_lora(args):
    """Use diffusers built-in LoRA support"""
    print("Setting up diffusers LoRA...")

    # Create pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    # Freeze all components
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Enable LoRA for UNet using diffusers built-in method
    from diffusers.loaders import UNet2DConditionLoadersMixin

    # Set up LoRA
    lora_attn_procs = {}
    for name, attn_processor in pipe.unet.attn_processors.items():
        # Create LoRA attention processor
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else pipe.unet.config.cross_attention_dim
        )

        if name.startswith("mid_block"):
            hidden_size = pipe.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(pipe.unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = pipe.unet.config.block_out_channels[block_id]

        # Use LoRAAttnProcessor
        from diffusers.models.attention_processor import LoRAAttnProcessor

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.lora_rank,
        )

    pipe.unet.set_attn_processor(lora_attn_procs)

    # Count trainable parameters
    trainable_params = []
    for name, param in pipe.unet.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            trainable_params.append(param)

    print(f"Trainable LoRA parameters: {sum(p.numel() for p in trainable_params):,}")

    # Move to device
    pipe = pipe.to(args.device)

    # Extract components
    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_name, subfolder="scheduler"
    )

    return tokenizer, text_encoder, vae, unet, noise_scheduler, trainable_params


# ============================================================================
# TRAINING FUNCTION
# ============================================================================


def train_simple(args):
    """Simple training without LoRA complexity"""
    print("Starting training...")

    # Setup models
    tokenizer, text_encoder, vae, unet, noise_scheduler, trainable_params = (
        setup_models_simple(args)
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
    )

    print(f"Training on {len(train_dataset)} samples")
    print(f"Batch size: {args.batch_size}")
    print(f"Total iterations per epoch: {len(train_dataloader)}")

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

            # Gradient clipping
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
                args.checkpoint_dir, f"unet_epoch_{epoch + 1}.pt"
            )

            # Save only trainable parameters
            state_dict = {}
            for name, param in unet.named_parameters():
                if param.requires_grad:
                    state_dict[name] = param.data

            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": state_dict,
                    "loss": avg_loss,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training completed!")

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "unet_final.pt")

    state_dict = {}
    for name, param in unet.named_parameters():
        if param.requires_grad:
            state_dict[name] = param.data

    torch.save({"state_dict": state_dict, "args": vars(args)}, final_path)
    print(f"Final model saved to {final_path}")

    return unet


def train_with_diffusers_lora(args):
    """Train using diffusers built-in LoRA"""
    print("Training with diffusers LoRA...")

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
                args.checkpoint_dir, f"lora_epoch_{epoch + 1}.safetensors"
            )

            # Save LoRA weights using diffusers method
            unet.save_attn_procs(
                args.checkpoint_dir, weight_name=f"lora_epoch_{epoch + 1}.safetensors"
            )
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training completed!")

    # Save final LoRA weights
    final_path = os.path.join(args.checkpoint_dir, "lora_final.safetensors")
    unet.save_attn_procs(args.checkpoint_dir, weight_name="lora_final.safetensors")
    print(f"Final LoRA weights saved to {final_path}")

    return unet


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate_model(model_path, args):
    """Evaluate trained model"""
    print(f"Evaluating model from {model_path}")

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    # Load LoRA weights
    if model_path.endswith(".safetensors"):
        pipe.unet.load_attn_procs(model_path)
    else:
        # Try to load as state dict
        state_dict = torch.load(model_path, map_location="cpu")
        if "state_dict" in state_dict:
            pipe.unet.load_state_dict(state_dict["state_dict"], strict=False)
        else:
            pipe.unet.load_state_dict(state_dict, strict=False)

    pipe = pipe.to(args.device)

    # Generate test images
    test_prompts = ["emoji 1", "emoji 2", "emoji 3", "emoji 4"]

    os.makedirs("./evaluation", exist_ok=True)

    for i, prompt in enumerate(test_prompts):
        image = pipe(prompt, num_inference_steps=50).images[0]
        image.save(f"./evaluation/test_{i:03d}.png")
        print(f"Generated: {prompt}")

    print("Evaluation complete!")


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train Stable Diffusion")

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
        "--use_lora",
        action="store_true",
        help="Use diffusers LoRA (otherwise train selected layers)",
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

    # Evaluation
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate instead of train"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to model for evaluation"
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

    # Run training or evaluation
    if args.evaluate:
        if not args.model_path:
            print("Error: --model_path required for evaluation")
            return
        evaluate_model(args.model_path, args)
    else:
        if args.use_lora:
            print("\nUsing diffusers built-in LoRA...")
            train_with_diffusers_lora(args)
        else:
            print("\nUsing simple training (selected layers)...")
            train_simple(args)


if __name__ == "__main__":
    main()
