#!/usr/bin/env python3
"""
Evaluation script for LoRA trained inpainting model - FIXED VERSION
With proper mask inversion (black = damage → white = inpaint)
"""

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageOps
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA model using dataset masks - FIXED with mask inversion"
    )

    # Required arguments
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to trained LoRA weights directory",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=True,
        help="Path to test dataset directory (must contain orig/ and masks/ folders)",
    )

    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Base model name",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Number of samples to evaluate (0 for all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution for generation (images will be resized to this)",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5, help="Guidance scale"
    )
    parser.add_argument(
        "--invert_masks",
        action="store_true",
        default=True,
        help="Invert masks (black=damage → white=inpaint). Default: True",
    )
    parser.add_argument(
        "--orig_folder",
        type=str,
        default="orig",
        help="Name of folder containing original images",
    )
    parser.add_argument(
        "--mask_folder",
        type=str,
        default="masks",
        help="Name of folder containing mask images",
    )
    parser.add_argument(
        "--erased_folder",
        type=str,
        default="erased",
        help="Name of folder containing erased images (for comparison only)",
    )
    parser.add_argument(
        "--lora_file",
        type=str,
        default="pytorch_lora_weights.safetensors",
        help="Name of LoRA weights file",
    )

    return parser.parse_args()


def load_test_samples_with_masks(
    test_dir,
    orig_folder="orig",
    mask_folder="masks",
    erased_folder="erased",
    num_samples=0,
    invert_masks=True,
):
    """Load test samples with their corresponding masks - FIXED VERSION

    Args:
        test_dir: Directory containing test data
        orig_folder: Folder with original images (input for inpainting)
        mask_folder: Folder with masks (black = damage, white = intact)
        erased_folder: Folder with erased images (for comparison)
        num_samples: Number of samples to load (0 for all)
        invert_masks: Whether to invert masks (black=damage → white=inpaint)
    """
    test_dir = Path(test_dir)

    # Check for ORIGINAL folder (this is our input for inpainting)
    orig_dir = test_dir / orig_folder
    if not orig_dir.exists():
        # Try alternative names
        for folder_name in ["orig", "orig_test", "original", "originals"]:
            potential_dir = test_dir / folder_name
            if potential_dir.exists():
                orig_dir = potential_dir
                break

        if not orig_dir.exists():
            raise ValueError(f"No original image folder found in {test_dir}")

    print(f"Using original folder: {orig_dir}")

    # Check for mask folder
    mask_dir = test_dir / mask_folder
    if not mask_dir.exists():
        # Try alternative names
        for folder_name in ["masks", "masks_test", "mask"]:
            potential_dir = test_dir / folder_name
            if potential_dir.exists():
                mask_dir = potential_dir
                break

        if not mask_dir.exists():
            raise ValueError(f"No mask folder found in {test_dir}")

    print(f"Using mask folder: {mask_dir}")
    if invert_masks:
        print("Note: Masks will be inverted (black=damage → white=inpaint)")

    # Check for erased folder (for comparison only)
    erased_dir = test_dir / erased_folder
    has_erased = erased_dir.exists()
    if has_erased:
        print(f"Found erased folder for comparison: {erased_dir}")

    # Load original images (these are our inputs for inpainting)
    orig_files = sorted(list(orig_dir.glob("*.png")))

    if num_samples > 0 and num_samples < len(orig_files):
        random.seed(42)
        orig_files = random.sample(orig_files, num_samples)

    samples = []

    for orig_path in tqdm(orig_files, desc="Loading test samples"):
        # Extract base name
        filename = orig_path.stem  # e.g., "6__val_m00"
        base_name = filename  # Original files don't have _erased suffix

        # Get emoji ID for prompt
        emoji_id = base_name.split("__")[0]
        prompt = f"a damaged picture of a single emoji that needs to be repaired"

        # Load ORIGINAL image (this is the input for inpainting)
        original_image = Image.open(orig_path).convert("RGB")

        # Load corresponding mask
        mask_filename = f"{base_name}_mask.png"
        mask_path = mask_dir / mask_filename

        if not mask_path.exists():
            # Try alternative naming
            alt_mask_filename = f"{base_name}.png"
            alt_mask_path = mask_dir / alt_mask_filename

            if alt_mask_path.exists():
                mask_path = alt_mask_path
            else:
                print(f"Warning: Mask not found for {base_name}")
                continue

        mask_image = Image.open(mask_path).convert("L")

        # INVERT THE MASK if needed
        # Your masks: black = damage (should inpaint), white = intact (should keep)
        # SD expects: white = inpaint, black = keep
        if invert_masks:
            mask_image = ImageOps.invert(mask_image)
            print(f"  Inverted mask for {base_name}")

        # Load erased image for comparison only (not for input)
        erased_image = None
        if has_erased:
            erased_filename = f"{base_name}_erased.png"
            erased_path = erased_dir / erased_filename
            if erased_path.exists():
                erased_image = Image.open(erased_path).convert("RGB")

        samples.append(
            {
                "prompt": prompt,
                "original_image": original_image,  # Full original image (INPUT)
                "mask_image": mask_image,  # Inverted if needed (white = inpaint)
                "erased_image": erased_image,  # For visualization comparison only
                "ground_truth": original_image,  # For metrics calculation
                "base_name": base_name,
                "emoji_id": emoji_id,
                "orig_filename": orig_path.name,
                "mask_filename": mask_path.name,
            }
        )

    print(f"Loaded {len(samples)} test samples with masks")

    # Show sample mask statistics
    if samples:
        sample_mask = np.array(samples[0]["mask_image"])
        black_pixels = np.sum(sample_mask < 128)
        white_pixels = np.sum(sample_mask >= 128)
        total_pixels = sample_mask.size
        print(
            f"Sample mask: {white_pixels / total_pixels * 100:.1f}% white (inpaint), "
            f"{black_pixels / total_pixels * 100:.1f}% black (keep)"
        )

    return samples


def calculate_inpainting_metrics(original, generated, mask):
    """Calculate inpainting-specific metrics"""
    if original is None or generated is None:
        return {}

    # Convert to numpy arrays
    orig_np = np.array(original).astype(np.float32) / 255.0
    gen_np = np.array(generated).astype(np.float32) / 255.0
    mask_np = np.array(mask).astype(np.float32) / 255.0

    # Get binary mask (masked regions where we should inpaint)
    # After inversion: white (>0.5) = inpaint, black (<0.5) = keep
    mask_binary = mask_np > 0.5

    if not mask_binary.any():
        print("Warning: Mask has no positive regions (nothing to inpaint)")
        return {}

    # Get unmasked regions (where mask == 0, should be preserved)
    unmasked_binary = ~mask_binary

    # 1. Reconstruction loss in masked regions (inpainting quality)
    orig_masked = orig_np[mask_binary]
    gen_masked = gen_np[mask_binary]

    if len(orig_masked) > 0:
        mse_masked = np.mean((orig_masked - gen_masked) ** 2)
        psnr_masked = (
            20 * np.log10(1.0 / np.sqrt(mse_masked)) if mse_masked > 0 else float("inf")
        )
    else:
        mse_masked = 0
        psnr_masked = float("inf")

    # 2. Preservation loss in unmasked regions (should not change)
    orig_unmasked = orig_np[unmasked_binary]
    gen_unmasked = gen_np[unmasked_binary]

    if len(orig_unmasked) > 0:
        mse_unmasked = np.mean((orig_unmasked - gen_unmasked) ** 2)
        psnr_unmasked = (
            20 * np.log10(1.0 / np.sqrt(mse_unmasked))
            if mse_unmasked > 0
            else float("inf")
        )
    else:
        mse_unmasked = 0
        psnr_unmasked = float("inf")

    # 3. Overall metrics
    mse_overall = np.mean((orig_np - gen_np) ** 2)
    psnr_overall = (
        20 * np.log10(1.0 / np.sqrt(mse_overall)) if mse_overall > 0 else float("inf")
    )

    # 4. Mask area ratio
    mask_area_ratio = np.sum(mask_binary) / mask_binary.size

    return {
        "mse_masked": float(mse_masked),
        "psnr_masked": float(psnr_masked),
        "mse_unmasked": float(mse_unmasked),
        "psnr_unmasked": float(psnr_unmasked),
        "mse_overall": float(mse_overall),
        "psnr_overall": float(psnr_overall),
        "mask_area_ratio": float(mask_area_ratio),
        "pixels_to_inpaint": int(np.sum(mask_binary)),
        "pixels_to_preserve": int(np.sum(unmasked_binary)),
    }


def create_inpainting_visualization(
    input_img,
    mask_img,
    generated_img,
    original_img=None,
    erased_img=None,
    metrics=None,
):
    """Create visualization for inpainting results"""

    num_plots = (
        3
        + (1 if original_img is not None else 0)
        + (1 if erased_img is not None else 0)
    )
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

    idx = 0

    # 1. Original Image (Input to inpainting)
    axes[idx].imshow(input_img)
    axes[idx].set_title("Input (Original Image)")
    axes[idx].axis("off")
    idx += 1

    # 2. Mask (white = inpaint, black = keep)
    axes[idx].imshow(mask_img, cmap="gray", vmin=0, vmax=255)
    axes[idx].set_title("Mask (White = Inpaint)")
    axes[idx].axis("off")
    idx += 1

    # 3. Generated result
    axes[idx].imshow(generated_img)
    axes[idx].set_title("Generated Inpainting")
    axes[idx].axis("off")
    idx += 1

    # 4. Erased image (for comparison)
    if erased_img is not None:
        axes[idx].imshow(erased_img)
        axes[idx].set_title("Erased Image")
        axes[idx].axis("off")
        idx += 1

    # 5. Ground truth (original again, for metrics)
    if original_img is not None:
        axes[idx].imshow(original_img)
        axes[idx].set_title("Ground Truth")
        axes[idx].axis("off")
        idx += 1

    # Add metrics as text
    if metrics:
        metrics_text = []
        if "psnr_masked" in metrics:
            metrics_text.append(f"PSNR (inpainted): {metrics['psnr_masked']:.2f} dB")
        if "psnr_unmasked" in metrics:
            metrics_text.append(f"PSNR (preserved): {metrics['psnr_unmasked']:.2f} dB")
        if "mask_area_ratio" in metrics:
            metrics_text.append(f"Mask area: {metrics['mask_area_ratio'] * 100:.1f}%")

        if metrics_text:
            fig.suptitle("\n".join(metrics_text), fontsize=12, y=0.95)

    plt.tight_layout()
    return fig


def load_pipeline_with_lora(
    model_name, lora_path, lora_file="pytorch_lora_weights.safetensors"
):
    """Load pipeline with LoRA weights"""
    print(f"Loading base model: {model_name}")
    print(f"Loading LoRA from: {lora_path}")

    # Load base pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    )

    # Check what LoRA files are available
    lora_dir = Path(lora_path)

    # Look for LoRA files
    possible_lora_files = [
        lora_file,
        "pytorch_lora_weights.safetensors",
        "pytorch_lora_weights.bin",
        "lora_weights.safetensors",
        "lora_weights.bin",
    ]

    lora_file_to_load = None
    for lora_file_name in possible_lora_files:
        lora_file_path = lora_dir / lora_file_name
        if lora_file_path.exists():
            lora_file_to_load = lora_file_path
            print(f"Found LoRA file: {lora_file_to_load}")
            break

    if lora_file_to_load is None:
        # Try to find any safetensors or bin files
        for ext in ["*.safetensors", "*.bin"]:
            found = list(lora_dir.glob(ext))
            if found:
                lora_file_to_load = found[0]
                print(f"Found LoRA file: {lora_file_to_load}")
                break

    if lora_file_to_load is None:
        print("Warning: No LoRA file found. Trying to load from directory...")
        lora_file_to_load = lora_path

    # Load LoRA weights
    if str(lora_file_to_load).endswith(".safetensors"):
        pipe.unet.load_attn_procs(str(lora_file_to_load), use_safetensors=True)
    else:
        pipe.unet.load_attn_procs(str(lora_file_to_load))

    return pipe


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pipeline with LoRA
    pipe = load_pipeline_with_lora(args.model_name, args.lora_path, args.lora_file)

    # Enable memory efficient attention if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Enabled xformers memory efficient attention")
    except:
        print("XFormers not available, using default attention")

    # Move to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (slow)")

    # Load test samples with masks (NOW USING ORIGINAL IMAGES AS INPUT)
    test_samples = load_test_samples_with_masks(
        args.test_data_dir,
        args.orig_folder,
        args.mask_folder,
        args.erased_folder,
        args.num_samples if args.num_samples > 0 else 0,
        args.invert_masks,
    )

    if not test_samples:
        print("No test samples found!")
        return

    all_metrics = []
    results = []

    print(f"\nEvaluating on {len(test_samples)} samples...")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")

    for i, sample in enumerate(tqdm(test_samples, desc="Processing samples")):
        try:
            # Resize images to target resolution
            # Use ORIGINAL image as input (not erased!)
            input_image = sample["original_image"].resize(
                (args.resolution, args.resolution), Image.Resampling.LANCZOS
            )
            mask_image = sample["mask_image"].resize(
                (args.resolution, args.resolution), Image.Resampling.NEAREST
            )

            # Generate inpainting
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                generated_image = pipe(
                    prompt=sample["prompt"],
                    image=input_image,  # ORIGINAL image with damage
                    mask_image=mask_image,  # INVERTED mask (white = inpaint)
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=torch.Generator(
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    ).manual_seed(args.seed + i),
                ).images[0]

            # Calculate metrics
            metrics = {}
            ground_truth_resized = sample["ground_truth"].resize(
                (args.resolution, args.resolution), Image.Resampling.LANCZOS
            )
            metrics = calculate_inpainting_metrics(
                ground_truth_resized, generated_image, mask_image
            )
            all_metrics.append(metrics)

            # Save individual files
            base_filename = f"{i:03d}_{sample['emoji_id']}_{sample['base_name']}"

            # Save generated result
            generated_image.save(
                os.path.join(args.output_dir, f"{base_filename}_generated.png")
            )

            # Save input and mask
            input_image.save(
                os.path.join(args.output_dir, f"{base_filename}_input.png")
            )
            mask_image.save(os.path.join(args.output_dir, f"{base_filename}_mask.png"))

            # Save ground truth
            sample["ground_truth"].save(
                os.path.join(args.output_dir, f"{base_filename}_ground_truth.png")
            )

            # Save erased image for comparison if available
            if sample["erased_image"] is not None:
                sample["erased_image"].save(
                    os.path.join(args.output_dir, f"{base_filename}_erased.png")
                )

            # Create and save visualization
            fig = create_inpainting_visualization(
                input_image,
                mask_image,
                generated_image,
                ground_truth_resized,
                sample["erased_image"],
                metrics,
            )
            fig.savefig(
                os.path.join(args.output_dir, f"{base_filename}_comparison.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

            # Store results
            results.append(
                {
                    "index": i,
                    "emoji_id": sample["emoji_id"],
                    "prompt": sample["prompt"],
                    "base_name": sample["base_name"],
                    "mask_inverted": args.invert_masks,
                    "metrics": metrics,
                }
            )

        except Exception as e:
            print(f"\nError processing sample {i} ({sample['base_name']}): {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    # Print summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if all_metrics:
        # Calculate average metrics
        summary = {}
        metric_names = all_metrics[0].keys()

        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            if values and not np.isinf(values).all():  # Skip if all values are inf
                valid_values = [v for v in values if not np.isinf(v)]
                if valid_values:
                    summary[f"{metric_name}_mean"] = float(np.mean(valid_values))
                    summary[f"{metric_name}_std"] = float(np.std(valid_values))
                    summary[f"{metric_name}_min"] = float(np.min(valid_values))
                    summary[f"{metric_name}_max"] = float(np.max(valid_values))

                    # Print formatted results
                    if "psnr" in metric_name:
                        print(
                            f"{metric_name.replace('_', ' ').title()}: "
                            f"{np.mean(valid_values):.2f} ± {np.std(valid_values):.2f} dB"
                        )
                    elif "mse" in metric_name:
                        print(
                            f"{metric_name.replace('_', ' ').title()}: "
                            f"{np.mean(valid_values):.6f} ± {np.std(valid_values):.6f}"
                        )
                    elif "ratio" in metric_name:
                        print(
                            f"{metric_name.replace('_', ' ').title()}: "
                            f"{np.mean(valid_values) * 100:.1f} ± {np.std(valid_values) * 100:.1f}%"
                        )
                    elif "pixels" in metric_name:
                        print(
                            f"{metric_name.replace('_', ' ').title()}: "
                            f"{np.mean(valid_values):.0f} ± {np.std(valid_values):.0f}"
                        )

        # Save detailed results
        results_file = os.path.join(args.output_dir, "evaluation_results.json")
        with open(results_file, "w") as f:
            json.dump(
                {
                    "config": vars(args),
                    "summary_statistics": summary,
                    "per_sample_results": results,
                    "all_metrics": all_metrics,
                },
                f,
                indent=2,
            )

        print(f"\nDetailed results saved to: {results_file}")

    # Create results grid
    create_results_grid(args.output_dir)

    print(f"\nAll visualizations saved to: {args.output_dir}")
    print("=" * 60)


def create_results_grid(output_dir):
    """Create a grid of generated images"""
    import glob

    from PIL import Image

    generated_images = sorted(glob.glob(os.path.join(output_dir, "*_generated.png")))

    if not generated_images:
        return

    # Load first image to get size
    sample_img = Image.open(generated_images[0])
    img_width, img_height = sample_img.size

    # Create grid (max 20 images)
    n = min(20, len(generated_images))
    grid_cols = min(5, n)
    grid_rows = (n + grid_cols - 1) // grid_cols

    grid = Image.new("RGB", (grid_cols * img_width, grid_rows * img_height))

    for i, img_path in enumerate(generated_images[:n]):
        img = Image.open(img_path)
        row = i // grid_cols
        col = i % grid_cols
        grid.paste(img, (col * img_width, row * img_height))

    grid.save(os.path.join(output_dir, "results_grid.png"))
    print(f"Results grid saved: {os.path.join(output_dir, 'results_grid.png')}")


if __name__ == "__main__":
    main()
