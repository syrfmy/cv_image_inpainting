#!/usr/bin/env python3
"""
Stable Diffusion LoRA Evaluation Script
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# Import from diffusers
from diffusers import (
    StableDiffusionPipeline, 
    DDIMScheduler,
    AutoencoderKL, 
    UNet2DConditionModel
)
from transformers import CLIPTokenizer, CLIPTextModel

from torch.utils.data import DataLoader
from train_lora import StableDiffusionDataset, LoRALayer, inject_lora_to_linear

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

def calculate_difference(original, generated):
    """Calculate absolute difference between original and generated images"""
    # Both images should be in [-1, 1] range
    diff = torch.abs(original - generated)
    
    # Normalize difference for visualization
    diff_normalized = diff / (diff.max() + 1e-8)
    
    # Apply colormap (convert to heatmap)
    diff_rgb = torch.zeros_like(original)
    diff_rgb[0] = diff_normalized.mean(dim=0)  # Red channel
    diff_rgb[1] = 0  # Green channel
    diff_rgb[2] = 1 - diff_normalized.mean(dim=0)  # Blue channel
    
    return diff_rgb

def tensor_to_pil(tensor):
    """Convert tensor to PIL image"""
    # Convert from [-1, 1] to [0, 255]
    if tensor.min() < 0:
        tensor = (tensor + 1) * 127.5
    else:
        tensor = tensor * 255
    
    tensor = tensor.clamp(0, 255).byte()
    tensor = tensor.cpu().permute(1, 2, 0).numpy()
    
    return Image.fromarray(tensor)

# ============================================================================
# RESULT SAVING
# ============================================================================

def save_results_as_dataset(eval_results, output_dir, save_visualizations=True):
    """Save evaluation results as organized dataset with metrics"""
    
    output_dir = Path(output_dir)
    
    # Create directory structure
    dirs = {
        'erased': output_dir / 'erased',
        'masks': output_dir / 'masks',
        'orig': output_dir / 'orig',
        'generated': output_dir / 'generated',
        'difference': output_dir / 'difference',
    }
    
    if save_visualizations:
        dirs['visualizations'] = output_dir / 'visualizations'
    
    for dir_path in dirs.values():
        if dir_path:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        'samples': [],
        'summary': {
            'total_samples': len(eval_results),
            'samples_with_original': sum(1 for r in eval_results if r['has_original']),
            'avg_psnr': 0,
            'avg_ssim': 0,
            'median_psnr': 0,
            'median_ssim': 0,
        }
    }
    
    psnr_values = []
    ssim_values = []
    
    for idx, result in enumerate(tqdm(eval_results, desc="Saving results")):
        base_filename = result['filename']
        
        # Save all images
        filepaths = {}
        
        # Save erased
        erased_pil = tensor_to_pil(result['erased'])
        erased_path = output_dir / 'erased' / f"{base_filename}_erased.png"
        erased_pil.save(erased_path)
        filepaths['erased'] = str(erased_path.relative_to(output_dir))
        
        # Save mask (convert to 3 channels for saving)
        mask_tensor = result['mask'].repeat(3, 1, 1)
        mask_pil = tensor_to_pil(mask_tensor)
        mask_path = output_dir / 'masks' / f"{base_filename}_mask.png"
        mask_pil.save(mask_path)
        filepaths['mask'] = str(mask_path.relative_to(output_dir))
        
        # Save original if exists
        if result['has_original']:
            original_pil = tensor_to_pil(result['original'])
            orig_path = output_dir / 'orig' / f"{base_filename}_orig.png"
            original_pil.save(orig_path)
            filepaths['original'] = str(orig_path.relative_to(output_dir))
        else:
            filepaths['original'] = None
        
        # Save generated
        generated_pil = tensor_to_pil(result['generated'])
        generated_path = output_dir / 'generated' / f"{base_filename}_generated.png"
        generated_pil.save(generated_path)
        filepaths['generated'] = str(generated_path.relative_to(output_dir))
        
        # Save difference if original exists
        if result['has_original'] and result['difference'] is not None:
            diff_pil = tensor_to_pil(result['difference'])
            diff_path = output_dir / 'difference' / f"{base_filename}_diff.png"
            diff_pil.save(diff_path)
            filepaths['difference'] = str(diff_path.relative_to(output_dir))
        else:
            filepaths['difference'] = None
        
        # Save visualization if requested
        if save_visualizations and result['has_original']:
            viz_path = output_dir / 'visualizations' / f"{base_filename}_viz.png"
            create_visualization(result, viz_path)
            filepaths['visualization'] = str(viz_path.relative_to(output_dir))
        
        # Add sample to metadata
        sample_metadata = {
            'id': idx,
            'filename': base_filename,
            'emoji_id': result['emoji_id'],
            'has_original': result['has_original'],
            'psnr': float(result['psnr']),
            'ssim': float(result['ssim']),
            'filepaths': filepaths,
        }
        
        metadata['samples'].append(sample_metadata)
        
        # Collect metrics for summary
        if result['has_original']:
            psnr_values.append(result['psnr'])
            ssim_values.append(result['ssim'])
    
    # Calculate summary statistics
    if psnr_values:
        metadata['summary']['avg_psnr'] = float(np.mean(psnr_values))
        metadata['summary']['avg_ssim'] = float(np.mean(ssim_values))
        metadata['summary']['median_psnr'] = float(np.median(psnr_values))
        metadata['summary']['median_ssim'] = float(np.median(ssim_values))
    
    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save summary as text
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total samples: {metadata['summary']['total_samples']}\n")
        f.write(f"Samples with original: {metadata['summary']['samples_with_original']}\n")
        if psnr_values:
            f.write(f"Average PSNR: {metadata['summary']['avg_psnr']:.2f} dB\n")
            f.write(f"Average SSIM: {metadata['summary']['avg_ssim']:.4f}\n")
            f.write(f"Median PSNR: {metadata['summary']['median_psnr']:.2f} dB\n")
            f.write(f"Median SSIM: {metadata['summary']['median_ssim']:.4f}\n")
        f.write(f"\nDataset saved to: {output_dir}\n")
    
    print(f"\nEvaluation results saved to: {output_dir}")
    print(f"  - Images: {len(eval_results)} samples")
    print(f"  - With originals: {metadata['summary']['samples_with_original']} samples")
    if psnr_values:
        print(f"  - Average PSNR: {metadata['summary']['avg_psnr']:.2f} dB")
        print(f"  - Average SSIM: {metadata['summary']['avg_ssim']:.4f}")
    
    return metadata

def create_visualization(result, save_path):
    """Create visualization grid showing all images"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    titles = ['Erased', 'Mask', 'Original', 'Generated', 'Difference', 'Metrics']
    
    # Convert tensors to numpy for display
    def tensor_to_display(tensor):
        if tensor is None:
            return None
        # Convert from [-1, 1] to [0, 1]
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1).cpu()
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        return tensor.numpy()
    
    images = [
        tensor_to_display(result['erased']),
        tensor_to_display(result['mask'].repeat(3, 1, 1)),  # Convert mask to RGB
        tensor_to_display(result['original']) if result['has_original'] else None,
        tensor_to_display(result['generated']),
        tensor_to_display(result['difference']) if result['has_original'] and result['difference'] is not None else None,
        None  # For metrics text
    ]
    
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        ax.set_title(titles[i])
        ax.axis('off')
        
        if i < 5 and images[i] is not None:
            ax.imshow(images[i])
        elif i == 5:  # Metrics display
            if result['has_original']:
                ax.text(0.5, 0.5, 
                       f"PSNR: {result['psnr']:.2f} dB\nSSIM: {result['ssim']:.4f}",
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            else:
                ax.text(0.5, 0.5, 
                       "No original\nfor comparison",
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
    
    plt.suptitle(f"Sample: {result['filename']} (emoji_id: {result['emoji_id']})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

@torch.no_grad()
def evaluate_model_full(args):
    """Run full evaluation inference"""
    
    print("Loading model...")
    
    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer")
    
    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained(args.model_name, subfolder="text_encoder")
    text_encoder.to(args.device)
    text_encoder.requires_grad_(False)
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae")
    vae.to(args.device)
    vae.requires_grad_(False)
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(args.model_name, subfolder="unet")
    unet.to(args.device)
    unet.requires_grad_(False)
    
    # Load LoRA weights if provided
    if args.model_path:
        print(f"Loading LoRA weights from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=args.device)
        
        # Inject LoRA first
        inject_lora_to_linear(unet, rank=args.lora_rank)
        
        # Load LoRA weights
        lora_state_dict = checkpoint.get('lora_state_dict', checkpoint)
        for name, param in unet.named_parameters():
            if 'lora' in name and name in lora_state_dict:
                param.data.copy_(lora_state_dict[name])
    
    # Noise scheduler
    noise_scheduler = DDIMScheduler.from_pretrained(args.model_name, subfolder="scheduler")
    
    # Load evaluation dataset
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
    
    print(f"Evaluating on {len(eval_dataset)} samples...")
    
    all_results = []
    
    for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Generating")):
        # Move to device
        original = batch['original'].to(args.device)
        erased = batch['erased'].to(args.device)
        mask = batch['mask'].to(args.device)
        input_ids = batch['input_ids'].to(args.device)
        has_original = batch['has_original']
        emoji_ids = batch['emoji_id']
        filenames = batch['filename']
        
        # Encode to latent space
        with torch.no_grad():
            latents_erased = vae.encode(erased).latent_dist.sample() * 0.18215
            
            # Get text embeddings
            if args.use_text_conditioning:
                encoder_hidden_states = text_encoder(input_ids)[0]
            else:
                encoder_hidden_states = None
        
        # Generate using DDIM
        noise_scheduler.set_timesteps(args.num_inference_steps)
        
        # Start from random noise
        latents = torch.randn_like(latents_erased)
        
        # Use mask to guide generation
        mask_latent = F.interpolate(mask, size=latents_erased.shape[-2:], mode='bilinear')
        latents = latents_erased * (1 - mask_latent) + latents * mask_latent
        
        # Denoising loop
        for t in noise_scheduler.timesteps:
            timestep = torch.full((latents.shape[0],), t, device=args.device, dtype=torch.long)
            noise_pred = unet(latents, timestep, encoder_hidden_states).sample
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents
        with torch.no_grad():
            generated = vae.decode(latents / 0.18215).sample
        
        # Calculate metrics for each sample
        for i in range(len(filenames)):
            result = {
                'filename': filenames[i],
                'emoji_id': emoji_ids[i],
                'has_original': has_original[i].item(),
                'erased': erased[i],
                'mask': mask[i],
                'original': original[i] if has_original[i] else None,
                'generated': generated[i],
                'difference': None,
                'psnr': 0.0,
                'ssim': 0.0,
            }
            
            # Calculate metrics if original exists
            if has_original[i]:
                result['psnr'] = calculate_psnr(original[i], generated[i])
                result['ssim'] = calculate_ssim(original[i].unsqueeze(0), generated[i].unsqueeze(0))
                
                # Calculate difference image
                result['difference'] = calculate_difference(original[i], generated[i])
            
            all_results.append(result)
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            valid_results = [r for r in all_results if r['has_original']]
            if valid_results:
                avg_psnr = np.mean([r['psnr'] for r in valid_results])
                avg_ssim = np.mean([r['ssim'] for r in valid_results])
                print(f"  Batch {batch_idx+1}/{len(eval_dataloader)} - "
                      f"Samples: {len(valid_results)} - "
                      f"PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
    
    return all_results

@torch.no_grad()
def evaluate_baseline(args):
    """Baseline evaluation (generated = erased)"""
    
    print("Running baseline evaluation (generated = erased)...")
    
    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer")
    
    # Load evaluation dataset
    eval_dataset = StableDiffusionDataset(
        data_path=args.eval_data,
        tokenizer=tokenizer,
        resolution=args.resolution,
        is_test=True
    )
    
    all_results = []
    
    for idx in tqdm(range(len(eval_dataset)), desc="Evaluating"):
        sample = eval_dataset[idx]
        
        # For baseline, generated = erased
        generated = sample['erased']
        
        result = {
            'filename': sample['filename'],
            'emoji_id': sample['emoji_id'],
            'has_original': sample['has_original'],
            'erased': sample['erased'],
            'mask': sample['mask'],
            'original': sample['original'] if sample['has_original'] else None,
            'generated': generated,
            'difference': None,
            'psnr': 0.0,
            'ssim': 0.0,
        }
        
        if sample['has_original']:
            result['psnr'] = calculate_psnr(sample['original'], generated)
            result['ssim'] = calculate_ssim(sample['original'].unsqueeze(0), generated.unsqueeze(0))
            result['difference'] = calculate_difference(sample['original'], generated)
        
        all_results.append(result)
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Stable Diffusion LoRA")
    
    # Dataset paths
    parser.add_argument("--eval_data", type=str, required=True,
                       help="Path to evaluation dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="Base model name")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to LoRA model weights (optional)")
    parser.add_argument("--lora_rank", type=int, default=4,
                       help="Rank for LoRA")
    parser.add_argument("--use_text_conditioning", action="store_true",
                       help="Use text conditioning")
    
    # Evaluation settings
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--resolution", type=int, default=64,
                       help="Image resolution")
    
    # Baseline evaluation
    parser.add_argument("--baseline", action="store_true",
                       help="Run baseline evaluation (generated = erased)")
    
    # Output settings
    parser.add_argument("--save_visualizations", action="store_true",
                       help="Save visualization images")
    parser.add_argument("--no_save_visualizations", action="store_false", dest="save_visualizations",
                       help="Don't save visualization images")
    parser.set_defaults(save_visualizations=True)
    
    # System settings
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of data loader workers")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--no_cuda", action="store_true",
                       help="Disable CUDA even if available")
    
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
    
    # Run evaluation
    if args.baseline:
        results = evaluate_baseline(args)
        output_dir = Path(args.output_dir) / "baseline"
    else:
        results = evaluate_model_full(args)
        if args.model_path:
            model_name = Path(args.model_path).stem
            output_dir = Path(args.output_dir) / model_name
        else:
            output_dir = Path(args.output_dir) / "pretrained"
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = save_results_as_dataset(results, output_dir, args.save_visualizations)
    
    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
