#!/usr/bin/env python3
"""
Stable Diffusion Fine-Tuning - Simple attention layer training
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
    UNet2DConditionModel,
)
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# ============================================================================
# DATASET
# ============================================================================


class StableDiffusionDataset(Dataset):
    """Dataset untuk fine-tuning Stable Diffusion"""

    def __init__(self, data_path, tokenizer, image_size=64, is_test=False):
        """
        Args:
            data_path: Path to dataset directory
            tokenizer: CLIP tokenizer
            image_size: Input image size
            is_test: If True, treat as test dataset where orig folder is optional
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.image_size = image_size
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
# SETUP MODELS - NO LoRA, JUST ATTENTION LAYER TRAINING
# ============================================================================


def setup_models_simple(args):
    """Simple setup - train only attention projection layers"""
    print("Setting up models for simple fine-tuning...")

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

    # Check UNet configuration
    print(f"UNet sample size: {unet.config.sample_size}")
    print(f"UNet in channels: {unet.config.in_channels}")

    # Freeze all UNet parameters first
    unet.requires_grad_(False)

    # Unfreeze only attention projection layers (like what LoRA would train)
    trainable_params = []

    # List of attention layer patterns to train
    attention_patterns = [
        "to_q",  # Query projection
        "to_k",  # Key projection
        "to_v",  # Value projection
        "to_out",  # Output projection
        "proj_out",  # Another output projection variant
        "proj_in",  # Input projection
    ]

    for name, param in unet.named_parameters():
        # Check if this parameter is part of an attention layer
        if any(pattern in name for pattern in attention_patterns):
            param.requires_grad = True
            trainable_params.append(param)
            # print(f"Training: {name}")  # Uncomment to see which layers are trained

    print(f"Training {len(trainable_params)} attention layers")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Count total parameters for comparison
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"Total UNet parameters: {total_params:,}")
    print(
        f"Training percentage: {sum(p.numel() for p in trainable_params) / total_params * 100:.2f}%"
    )

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_name, subfolder="scheduler"
    )

    return tokenizer, text_encoder, vae, unet, noise_scheduler, trainable_params


# ============================================================================
# TRAINING FUNCTION - SIMPLE NO LoRA
# ============================================================================


def train_simple(args):
    """Simple training without LoRA complexity"""
    print("Starting simple fine-tuning (attention layers only)...")

    # Setup models
    tokenizer, text_encoder, vae, unet, noise_scheduler, trainable_params = (
        setup_models_simple(args)
    )

    # Training dataset
    train_dataset = StableDiffusionDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        image_size=args.image_size,
        is_test=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    print(f"Training on {len(train_dataset)} samples")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(
        f"Latent size: {args.image_size // 8}x{args.image_size // 8} (downsample factor 8)"
    )

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    # Learning rate scheduler (optional)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_dataloader),
        eta_min=args.learning_rate * 0.1,
    )

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Save configuration
    config_file = os.path.join(args.checkpoint_dir, "config.txt")
    with open(config_file, "w") as f:
        f.write("Training Configuration:\n")
        f.write("=" * 50 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write(
            f"\nTrainable parameters: {sum(p.numel() for p in trainable_params):,}\n"
        )
        f.write(f"Image size: {args.image_size}x{args.image_size}\n")
        f.write(f"Dataset size: {len(train_dataset)} samples\n")

    # Training loop
    unet.train()
    best_loss = float("inf")

    # Track training history
    train_loss_history = []
    lr_history = []

    for epoch in range(args.epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            original = batch["original"].to(args.device)
            input_ids = batch["input_ids"].to(args.device)

            # Encode images to latent space
            with torch.no_grad():
                # Encode to latents
                latents_original = vae.encode(original).latent_dist.sample()

                # Scale latents (standard SD practice)
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

            # Predict noise
            noise_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
            ).sample

            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            optimizer.step()
            scheduler.step()

            # Update metrics
            epoch_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            lr_history.append(current_lr)

            # Print debug info for first batch of first epoch
            if batch_idx == 0 and epoch == 0:
                print(f"\nDebug info (first batch):")
                print(f"  Original shape: {original.shape}")
                print(f"  Latents shape: {latents_original.shape}")
                print(f"  Noise pred shape: {noise_pred.shape}")
                print(f"  Loss: {loss.item():.4f}")

            progress_bar.set_postfix({"loss": loss.item(), "lr": f"{current_lr:.2e}"})

        avg_loss = epoch_loss / len(train_dataloader)
        train_loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

        # Save checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"
            )

            # Save model state
            torch.save(
                {
                    "epoch": epoch,
                    "unet_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "args": vars(args),
                    "train_loss_history": train_loss_history,
                    "lr_history": lr_history,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "unet_state_dict": unet.state_dict(),
                        "loss": avg_loss,
                        "args": vars(args),
                    },
                    best_path,
                )
                print(f"Best model saved (loss: {avg_loss:.4f})")

    print("Training completed!")

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    torch.save(
        {
            "epoch": args.epochs,
            "unet_state_dict": unet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "args": vars(args),
            "train_loss_history": train_loss_history,
            "lr_history": lr_history,
        },
        final_path,
    )
    print(f"Final model saved to {final_path}")

    # Plot training history
    if args.plot_training_history:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(lr_history[: len(lr_history) // 10])  # Plot first 10% of LR history
        plt.title("Learning Rate Schedule")
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(args.checkpoint_dir, "training_history.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Training history plot saved to {plot_path}")

        if args.show_plots:
            plt.show()

    return unet


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================


def evaluate_model(model_path, args):
    """Evaluate trained model"""
    print(f"Evaluating model from {model_path}")

    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer")

    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(args.model_name, subfolder="unet")
    unet.to(args.device)

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=args.device)
    unet.load_state_dict(checkpoint["unet_state_dict"])

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Training loss: {checkpoint.get('loss', 'unknown'):.4f}")

    # You can add evaluation logic here
    # For example, generate some test images

    return unet


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion (No LoRA)")

    # Dataset paths
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training dataset"
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="Path to evaluation dataset (optional)",
    )

    # Model settings
    parser.add_argument(
        "--model_name",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base model name",
    )
    parser.add_argument(
        "--use_text_conditioning", action="store_true", help="Use text conditioning"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Image size for training (default: 64)",
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

    # Visualization
    parser.add_argument(
        "--plot_training_history", action="store_true", help="Plot training history"
    )
    parser.add_argument(
        "--show_plots", action="store_true", help="Show plots during training"
    )

    # Evaluation mode
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
        train_simple(args)


if __name__ == "__main__":
    main()
