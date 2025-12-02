#!/usr/bin/env python3
"""
Stable Diffusion LoRA Training Script
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import from diffusers
from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionPipeline
)
from transformers import CLIPTokenizer, CLIPTextModel

from torch.utils.data import Dataset, DataLoader

# ============================================================================
# IMPLEMENTASI LoRA - FIXED VERSION
# ============================================================================

class LoRALinearLayer(nn.Module):
    """LoRA Linear Layer - Fixed implementation"""
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # Initialize LoRA matrices
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Initialize with small random values
        nn.init.normal_(self.lora_down.weight, std=1/rank)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        # Apply LoRA: x + alpha * (lora_up(lora_down(x)))
        return self.lora_up(self.lora_down(x)) * (self.alpha / self.rank)

def inject_lora_to_unet(unet, rank=4, target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]):
    """Inject LoRA into UNet with proper handling"""
    lora_params = []

    # Count total parameters before injection
    total_params_before = sum(p.numel() for p in unet.parameters())

    for name, module in unet.named_modules():
        # Check if this is a target module
        is_target = any(target in name for target in target_modules)

        # Only inject into Linear layers
        if is_target and isinstance(module, nn.Linear):
            # Create LoRA layer
            lora_layer = LoRALinearLayer(
                module.in_features,
                module.out_features,
                rank=rank
            ).to(module.weight.device)

            # Store original forward method
            original_forward = module.forward

            # Define new forward with LoRA
            def new_forward(x, original_forward=original_forward, lora_layer=lora_layer):
                return original_forward(x) + lora_layer(x)

            # Replace forward method
            module.forward = new_forward.__get__(module, nn.Linear)

            # Add LoRA parameters to list
            lora_params.extend(list(lora_layer.parameters()))

            # Freeze original weights
            module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)

    # Count LoRA parameters
    lora_param_count = sum(p.numel() for p in lora_params)
    total_params_after = total_params_before + lora_param_count

    print(f"LoRA parameters injected: {lora_param_count:,}")
    print(f"Total parameters before: {total_params_before:,}")
    print(f"Total parameters after: {total_params_after:,}")
    print(f"LoRA percentage: {lora_param_count/total_params_before*100:.2f}%")

    return lora_params

# Alternative: Use diffusers' built-in LoRA
def inject_lora_using_diffusers(unet, rank=4):
    """Inject LoRA using diffusers' built-in functionality"""
    from diffusers.loaders import LoraLoaderMixin
    from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0

    # Set all attention processors to LoRA
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if torch.__version__ >= "2.0.0":
            attn_processor_class = LoRAAttnProcessor2_0
        else:
            attn_processor_class = LoRAAttnProcessor

        lora_attn_procs[name] = attn_processor_class(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )

    unet.set_attn_processor(lora_attn_procs)

    # Collect LoRA parameters
    lora_params = []
    for name, proc in unet.attn_processors.items():
        if hasattr(proc, "to_q_lora"):
            lora_params.extend(list(proc.to_q_lora.parameters()))
            lora_params.extend(list(proc.to_k_lora.parameters()))
            lora_params.extend(list(proc.to_v_lora.parameters()))
            lora_params.extend(list(proc.to_out_lora.parameters()))

    print(f"LoRA parameters injected using diffusers: {sum(p.numel() for p in lora_params):,}")

    return lora_params

# ============================================================================
# DATASET
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
            assert self.orig_dir.exists(), f"Folder {self.orig_dir} tidak ditemukan untuk training dataset"

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
                    self.image_files.append({
                        'original': orig_path,
                        'erased': erased_path,
                        'mask': mask_path,
                        'emoji_id': emoji_id,
                        'filename': base_name,
                        'has_original': True
                    })
                else:
                    # Untuk test tanpa original, kita masih bisa gunakan erased sebagai placeholder
                    self.image_files.append({
                        'original': None,
                        'erased': erased_path,
                        'mask': mask_path,
                        'emoji_id': emoji_id,
                        'filename': base_name,
                        'has_original': False
                    })
            else:
                # Untuk training, orig harus ada
                if not orig_path.exists():
                    print(f"Warning: Original file {orig_path} tidak ditemukan untuk training dataset")
                    continue

                self.image_files.append({
                    'original': orig_path,
                    'erased': erased_path,
                    'mask': mask_path,
                    'emoji_id': emoji_id,
                    'filename': base_name,
                    'has_original': True
                })

        print(f"Found {len(self.image_files)} samples in {data_path}")
        if len(self.image_files) > 0:
            sample = self.image_files[0]
            print(f"Example: emoji_id={sample['emoji_id']}, filename={sample['filename']}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        item = self.image_files[idx]

        # Load images
        erased = Image.open(item['erased']).convert('RGB')
        mask = Image.open(item['mask']).convert('L')

        # Load original or use erased as placeholder
        if item['has_original']:
            original = Image.open(item['original']).convert('RGB')
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
        emoji_id = item['emoji_id']
        prompt = f"emoji {emoji_id}"

        # Tokenize prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            'original': original,
            'erased': erased,
            'mask': mask,
            'input_ids': text_inputs.input_ids[0],
            'emoji_id': emoji_id,
            'has_original': item['has_original'],
            'filename': item['filename']
        }

# ============================================================================
# METRICS
# ============================================================================

def calculate_psnr(original, generated):
    """Calculate PSNR between original and generated images"""
    mse = torch.mean((original - generated) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 2.0  # Since images are normalized to [-1, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(original, generated, window_size=11, size_average=True):
    """Calculate SSIM between original and generated images"""
    from math import exp

    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    # Reshape from [-1, 1] to [0, 1] for SSIM calculation
    original = (original + 1) / 2
    generated = (generated + 1) / 2

    (_, channel, height, width) = original.size()
    window = create_window(window_size, channel).to(original.device)

    mu1 = F.conv2d(original, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(generated, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(original*original, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(generated*generated, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(original*generated, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()

# ============================================================================
# TRAINING SETUP
# ============================================================================

def setup_models(args):
    """Load pre-trained models"""
    print("Loading models...")

    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer")

    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained(args.model_name, subfolder="text_encoder")
    text_encoder.to(args.device)
    text_encoder.requires_grad_(False)  # Freeze text encoder

    # Load VAE
    vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae")
    vae.to(args.device)
    vae.requires_grad_(False)  # Freeze VAE

    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(args.model_name, subfolder="unet")
    unet.to(args.device)

    # Inject LoRA - Use diffusers built-in method for stability
    print(f"Injecting LoRA with rank={args.lora_rank}...")
    try:
        # Try using diffusers built-in method first
        lora_params = inject_lora_using_diffusers(unet, rank=args.lora_rank)
        print("Using diffusers built-in LoRA injection")
    except:
        # Fall back to custom implementation
        print("Using custom LoRA injection")
        lora_params = inject_lora_to_unet(unet, rank=args.lora_rank)

    # Freeze UNet parameters (except LoRA)
    for name, param in unet.named_parameters():
        if 'lora' not in name:
            param.requires_grad_(False)

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name, subfolder="scheduler")

    return tokenizer, text_encoder, vae, unet, noise_scheduler, lora_params

@torch.no_grad()
def evaluate_model(unet, vae, text_encoder, eval_dataloader, noise_scheduler, device, args):
    """Evaluate model on evaluation dataset"""
    unet.eval()
    text_encoder.eval()

    total_psnr = 0
    total_ssim = 0
    num_samples = 0

    progress_bar = tqdm(eval_dataloader, desc="Evaluating")

    for batch in progress_bar:
        # Skip if no original images
        if not batch['has_original'].any():
            continue

        # Move to device
        original = batch['original'].to(device)
        erased = batch['erased'].to(device)
        mask = batch['mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        has_original = batch['has_original']

        # Filter only samples with original images
        idx_with_original = torch.where(has_original)[0]
        if len(idx_with_original) == 0:
            continue

        original = original[idx_with_original]
        erased = erased[idx_with_original]
        mask = mask[idx_with_original]
        input_ids = input_ids[idx_with_original]

        # Encode to latent space
        latents_original = vae.encode(original).latent_dist.sample() * 0.18215
        latents_erased = vae.encode(erased).latent_dist.sample() * 0.18215

        # Get text embeddings
        if args.use_text_conditioning:
            encoder_hidden_states = text_encoder(input_ids)[0]
        else:
            encoder_hidden_states = None

        # Generate using DDIM for faster inference
        noise_scheduler.set_timesteps(args.num_inference_steps)

        # Start from random noise
        latents = torch.randn_like(latents_erased)

        # Use mask to guide generation
        mask_latent = F.interpolate(mask, size=latents_erased.shape[-2:], mode='bilinear')
        latents = latents_erased * (1 - mask_latent) + latents * mask_latent

        # Denoising loop
        for t in noise_scheduler.timesteps:
            timestep = torch.full((latents.shape[0],), t, device=device, dtype=torch.long)
            noise_pred = unet(latents, timestep, encoder_hidden_states).sample
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        with torch.no_grad():
            generated = vae.decode(latents / 0.18215).sample

        # Calculate metrics
        for i in range(generated.shape[0]):
            psnr_val = calculate_psnr(original[i], generated[i])
            ssim_val = calculate_ssim(original[i].unsqueeze(0), generated[i].unsqueeze(0))

            total_psnr += psnr_val
            total_ssim += ssim_val
            num_samples += 1

        progress_bar.set_postfix({
            'PSNR': total_psnr / max(num_samples, 1),
            'SSIM': total_ssim / max(num_samples, 1)
        })

    # Calculate averages
    avg_psnr = total_psnr / max(num_samples, 1)
    avg_ssim = total_ssim / max(num_samples, 1)

    unet.train()
    text_encoder.train()

    return avg_psnr, avg_ssim

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args):
    """Main training function"""

    # Setup models
    tokenizer, text_encoder, vae, unet, noise_scheduler, lora_params = setup_models(args)

    # Training dataset
    train_dataset = StableDiffusionDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        resolution=args.resolution,
        is_test=False
    )

    # Evaluation dataset
    if args.eval_data:
        eval_dataset = StableDiffusionDataset(
            data_path=args.eval_data,
            tokenizer=tokenizer,
            resolution=args.resolution,
            is_test=True
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    else:
        eval_dataloader = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # Optimizer - only optimize LoRA parameters
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training metrics history
    train_loss_history = []
    eval_psnr_history = []
    eval_ssim_history = []

    # Training loop
    unet.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            original = batch['original'].to(args.device)
            erased = batch['erased'].to(args.device)
            mask = batch['mask'].to(args.device)
            input_ids = batch['input_ids'].to(args.device)

            # Encode images to latent space
            with torch.no_grad():
                latents_original = vae.encode(original).latent_dist.sample() * 0.18215
                latents_erased = vae.encode(erased).latent_dist.sample() * 0.18215

                # Get text embeddings
                if args.use_text_conditioning:
                    encoder_hidden_states = text_encoder(input_ids)[0]
                else:
                    encoder_hidden_states = None

            # Sample noise
            noise = torch.randn_like(latents_original)

            # Sample timestep
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents_original.shape[0],),
                device=args.device
            ).long()

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents_original, noise, timesteps)

            # Predict noise - FIXED: Ensure we're using the right dimensions
            # Get the noise prediction
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample

            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            global_step += 1

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(train_dataloader)
        train_loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        # Evaluate model
        if eval_dataloader and (epoch + 1) % args.eval_every_n_epochs == 0:
            print(f"\nEvaluating model at epoch {epoch+1}...")
            avg_psnr, avg_ssim = evaluate_model(
                unet, vae, text_encoder, eval_dataloader, noise_scheduler, args.device, args
            )
            eval_psnr_history.append(avg_psnr)
            eval_ssim_history.append(avg_ssim)
            print(f"Evaluation Metrics - PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}\n")

        # Save checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"lora_epoch_{epoch+1}.pt")

            # Save LoRA weights properly
            lora_state_dict = {}
            for name, param in unet.named_parameters():
                if 'lora' in name and param.requires_grad:
                    lora_state_dict[name] = param.data

            checkpoint_data = {
                'epoch': epoch,
                'lora_state_dict': lora_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'train_loss_history': train_loss_history,
                'eval_psnr_history': eval_psnr_history if eval_psnr_history else [],
                'eval_ssim_history': eval_ssim_history if eval_ssim_history else [],
                'args': vars(args)
            }

            torch.save(checkpoint_data, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training completed!")

    # Final evaluation
    if eval_dataloader:
        print("\nFinal evaluation...")
        final_psnr, final_ssim = evaluate_model(
            unet, vae, text_encoder, eval_dataloader, noise_scheduler, args.device, args
        )
        print(f"Final Metrics - PSNR: {final_psnr:.2f} dB, SSIM: {final_ssim:.4f}")
        eval_psnr_history.append(final_psnr)
        eval_ssim_history.append(final_ssim)

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "lora_final.pt")

    # Save LoRA weights properly
    lora_state_dict = {}
    for name, param in unet.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_state_dict[name] = param.data

    final_data = {
        'lora_state_dict': lora_state_dict,
        'train_loss_history': train_loss_history,
        'eval_psnr_history': eval_psnr_history if eval_psnr_history else [],
        'eval_ssim_history': eval_ssim_history if eval_ssim_history else [],
        'args': vars(args)
    }

    torch.save(final_data, final_path)
    print(f"Final LoRA weights saved to {final_path}")

    # Plot training history
    if args.plot_training_history:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot training loss
        ax1 = axes[0] if len(axes) > 1 else axes
        ax1.plot(train_loss_history)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        # Plot PSNR if available
        if eval_psnr_history and len(axes) > 1:
            ax2 = axes[1]
            eval_epochs = range(0, len(eval_psnr_history)*args.eval_every_n_epochs, args.eval_every_n_epochs)
            ax2.plot(eval_epochs[:len(eval_psnr_history)], eval_psnr_history)
            ax2.set_title('Evaluation PSNR')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('PSNR (dB)')
            ax2.grid(True)

        # Plot SSIM if available
        if eval_ssim_history and len(axes) > 2:
            ax3 = axes[2]
            eval_epochs = range(0, len(eval_ssim_history)*args.eval_every_n_epochs, args.eval_every_n_epochs)
            ax3.plot(eval_epochs[:len(eval_ssim_history)], eval_ssim_history)
            ax3.set_title('Evaluation SSIM')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('SSIM')
            ax3.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(args.checkpoint_dir, 'training_history.png'), dpi=150)
        if not args.no_show_plot:
            plt.show()

    print(f"\nTraining completed successfully!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train Stable Diffusion with LoRA")

    # Dataset paths
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training dataset")
    parser.add_argument("--eval_data", type=str, default=None,
                       help="Path to evaluation dataset (optional)")

    # Model settings
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="Base model name")
    parser.add_argument("--lora_rank", type=int, default=4,
                       help="Rank for LoRA (4-16 typically)")
    parser.add_argument("--use_text_conditioning", action="store_true",
                       help="Use text conditioning")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--resolution", type=int, default=64,
                       help="Image resolution")

    # Checkpoint settings
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--save_every_n_epochs", type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--eval_every_n_epochs", type=int, default=5,
                       help="Evaluate every N epochs")

    # Evaluation settings
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps for evaluation")

    # System settings
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of data loader workers")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--no_cuda", action="store_true",
                       help="Disable CUDA even if available")

    # Visualization
    parser.add_argument("--plot_training_history", action="store_true",
                       help="Plot training history")
    parser.add_argument("--no_show_plot", action="store_true",
                       help="Don't show plot, only save")

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

    # Run training
    train(args)

if __name__ == "__main__":
    main()
