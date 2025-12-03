#!/usr/bin/env python3
"""
Complete Evaluation Script for LoRA Model
FIXED VERSION - Correct mask inversion
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def calculate_metrics(img1, img2):
    """Calculate metrics between two images - FIXED for shape matching"""
    # Ensure both images are the same size
    if img1.size != img2.size:
        # Resize img2 to match img1
        img2 = img2.resize(img1.size, Image.LANCZOS)

    # Convert to numpy arrays
    arr1 = np.array(img1).astype(float)
    arr2 = np.array(img2).astype(float)

    # Check shapes match
    if arr1.shape != arr2.shape:
        raise ValueError(f"Shape mismatch after resizing: {arr1.shape} vs {arr2.shape}")

    metrics = {}

    # PSNR
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        metrics["psnr"] = float("inf")
    else:
        max_pixel = 255.0
        metrics["psnr"] = 20 * np.log10(max_pixel / np.sqrt(mse))

    # SSIM
    try:
        from skimage.metrics import structural_similarity as ssim

        if len(arr1.shape) == 3:
            # Convert to grayscale for SSIM
            arr1_gray = np.mean(arr1, axis=2)
            arr2_gray = np.mean(arr2, axis=2)
            metrics["ssim"] = ssim(arr1_gray, arr2_gray, data_range=255)
        else:
            metrics["ssim"] = ssim(arr1, arr2, data_range=255)
    except ImportError:
        metrics["ssim"] = 0.0
        print("Warning: skimage not installed, SSIM set to 0")

    # L1/L2 distances
    metrics["l1"] = np.mean(np.abs(arr1 - arr2))
    metrics["l2"] = np.sqrt(mse)

    return metrics


def invert_mask(mask_image):
    """Safely invert a mask image: black->white, white->black"""
    mask_array = np.array(mask_image)
    # Invert: 0->255, 255->0, etc.
    inverted_array = 255 - mask_array
    # Convert back to PIL
    return Image.fromarray(inverted_array).convert("L")


def evaluate_all_samples(args):
    """Evaluate model on ALL test samples - FIXED for consistent resolution and mask inversion"""
    print("=" * 60)
    print("COMPLETE EVALUATION - ALL SAMPLES")
    print("=" * 60)

    # IMPORTANT: Set the resolution based on training
    # If you trained on 64x64, use 64x64 for evaluation
    TARGET_SIZE = (args.resolution, args.resolution)
    print(f"Target resolution: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("Loading pipeline...")
    # Load base pipeline with correct parameters
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Load LoRA weights using the new method
    print(f"Loading LoRA weights from {args.lora_path}")
    try:
        # Try the new load_lora_weights method
        if hasattr(pipe, "load_lora_weights"):
            pipe.load_lora_weights(args.lora_path)
            print("Loaded LoRA weights using load_lora_weights()")
        else:
            # Fall back to old method
            pipe.unet.load_attn_procs(args.lora_path)
            print("Loaded LoRA weights using load_attn_procs()")
    except Exception as e:
        print(f"Error loading LoRA weights: {e}")
        # Try loading from safetensors directly
        try:
            import safetensors.torch

            safetensors_path = os.path.join(
                args.lora_path, "pytorch_lora_weights.safetensors"
            )
            if os.path.exists(safetensors_path):
                state_dict = safetensors.torch.load_file(safetensors_path, device="cpu")
                pipe.unet.load_state_dict(state_dict, strict=False)
                print("Loaded from safetensors")
            else:
                print("Warning: Could not load LoRA weights from safetensors")
        except ImportError:
            print("Warning: safetensors not installed")

    # Move to device
    pipe = pipe.to(device)

    # Enable memory efficient attention if available
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
        except:
            pass

    # Find ALL test images
    test_data_path = Path(args.test_data_dir)
    erased_dir = test_data_path / "erased"
    masks_dir = test_data_path / "masks"
    orig_dir = test_data_path / "orig"

    if not all([erased_dir.exists(), masks_dir.exists(), orig_dir.exists()]):
        print("Error: Missing required directories")
        print(f"  erased exists: {erased_dir.exists()}")
        print(f"  masks exists: {masks_dir.exists()}")
        print(f"  orig exists: {orig_dir.exists()}")
        return

    # Get ALL erased files
    test_files = list(erased_dir.glob("*_erased.png"))
    if not test_files:
        # Try other patterns
        test_files = list(erased_dir.glob("*.png"))

    if not test_files:
        print(f"No test files found in {erased_dir}")
        return

    print(f"Found {len(test_files)} samples to evaluate")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    composites_dir = output_dir / "composites"
    composites_dir.mkdir(exist_ok=True)

    if args.save_individual:
        generated_dir = output_dir / "generated"
        generated_dir.mkdir(exist_ok=True)
        input_dir = output_dir / "input"
        input_dir.mkdir(exist_ok=True)
        mask_dir = output_dir / "masks"
        mask_dir.mkdir(exist_ok=True)
        original_dir = output_dir / "originals"
        original_dir.mkdir(exist_ok=True)

    # Store all results
    all_results = []
    failed_samples = []

    # Progress bar for all samples
    progress_bar = tqdm(test_files, desc="Evaluating", unit="sample")

    for test_file in progress_bar:
        filename = test_file.stem

        # Extract base name
        if "_erased" in filename:
            base_name = filename.replace("_erased", "")
        else:
            base_name = filename

        # Find corresponding mask and original
        mask_file = masks_dir / f"{base_name}_mask.png"
        if not mask_file.exists():
            mask_file = masks_dir / f"{base_name}.png"

        orig_file = orig_dir / f"{base_name}.png"
        if not orig_file.exists():
            orig_file = orig_dir / f"{filename}.png"

        if not mask_file.exists():
            failed_samples.append(
                {"filename": filename, "reason": f"Missing mask file: {mask_file}"}
            )
            continue
        if not orig_file.exists():
            failed_samples.append(
                {"filename": filename, "reason": f"Missing original file: {orig_file}"}
            )
            continue

        try:
            # Load images
            image = Image.open(test_file).convert("RGB")
            mask_image = Image.open(mask_file).convert("L")
            original = Image.open(orig_file).convert("RGB")
        except Exception as e:
            failed_samples.append(
                {"filename": filename, "reason": f"Error loading images: {e}"}
            )
            continue

        # DEBUG: Print original sizes and mask info for first sample
        if len(all_results) == 0:
            print(f"\nDebug - First sample ({filename}):")
            print(f"  Input image: {image.size}")
            print(f"  Mask image: {mask_image.size}")
            print(f"  Original image: {original.size}")

            # Check mask values
            mask_array = np.array(mask_image)
            unique_vals = np.unique(mask_array)
            print(f"  Mask unique values: {unique_vals}")
            print(f"  Mask min: {mask_array.min()}, max: {mask_array.max()}")
            print(f"  Mask mean: {mask_array.mean():.1f}")

        # Resize ALL images to target resolution
        image = image.resize(TARGET_SIZE, Image.LANCZOS)
        mask_image = mask_image.resize(TARGET_SIZE, Image.NEAREST)
        original_resized = original.resize(TARGET_SIZE, Image.LANCZOS)

        # INVERT THE MASK for Stable Diffusion
        # If your training used black=damage, you need to invert for SD
        mask_image_inverted = invert_mask(mask_image)

        # DEBUG: Check inverted mask for first sample
        if len(all_results) == 0:
            inv_array = np.array(mask_image_inverted)
            print(f"  Inverted mask min: {inv_array.min()}, max: {inv_array.max()}")
            print(f"  Inverted mask mean: {inv_array.mean():.1f}")

        # Use training prompt (make sure this matches your training prompt!)
        prompt = "a damaged picture of a single emoji that needs to be repaired"

        # Generate - use INVERTED mask
        try:
            result = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask_image_inverted,  # Use INVERTED mask
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device).manual_seed(args.seed)
                if args.seed
                else None,
                height=TARGET_SIZE[1],
                width=TARGET_SIZE[0],
            ).images[0]
        except Exception as e:
            failed_samples.append(
                {"filename": filename, "reason": f"Error during generation: {e}"}
            )
            continue

        # DEBUG: Check generated image size for first sample
        if len(all_results) == 0:
            print(f"  Generated image: {result.size}")

        # Ensure generated image is correct size (resize if needed)
        if result.size != TARGET_SIZE:
            print(
                f"Warning: Generated image size {result.size} doesn't match target {TARGET_SIZE}"
            )
            result = result.resize(TARGET_SIZE, Image.LANCZOS)

        # Calculate metrics - use resized original
        metrics = calculate_metrics(result, original_resized)

        # Save individual images if requested
        if args.save_individual:
            result.save(generated_dir / f"{filename}_generated.png")
            image.save(input_dir / f"{filename}_input.png")
            mask_image.save(mask_dir / f"{filename}_mask.png")
            mask_image_inverted.save(mask_dir / f"{filename}_mask_inverted.png")
            original_resized.save(original_dir / f"{filename}_original.png")

        # Create and save composite with labels
        composite = Image.new("RGB", (TARGET_SIZE[0] * 4, TARGET_SIZE[1]))

        # Paste images
        composite.paste(image, (0, 0))
        composite.paste(mask_image.convert("RGB"), (TARGET_SIZE[0], 0))
        composite.paste(original_resized, (TARGET_SIZE[0] * 2, 0))
        composite.paste(result, (TARGET_SIZE[0] * 3, 0))

        # Add text labels if space permits
        try:
            draw = ImageDraw.Draw(composite)
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            labels = ["Input", "Mask", "Original", "Generated"]
            for i, label in enumerate(labels):
                draw.text(
                    (i * TARGET_SIZE[0] + 10, 10),
                    label,
                    fill="white",
                    stroke_width=2,
                    stroke_fill="black",
                    font=font,
                )
        except:
            pass  # Skip text if it fails

        composite_path = composites_dir / f"{filename}_composite.png"
        composite.save(composite_path)

        # Store result
        sample_result = {
            "filename": filename,
            "metrics": metrics,
            "composite_path": str(composite_path),
            "input_size": image.size,
            "generated_size": result.size,
            "original_size": original.size,
        }
        all_results.append(sample_result)

        # Update progress bar with current metrics
        progress_bar.set_postfix(
            {"PSNR": f"{metrics['psnr']:.1f}", "SSIM": f"{metrics['ssim']:.3f}"}
        )

    # Calculate overall statistics
    if all_results:
        metrics_names = ["psnr", "ssim", "l1", "l2"]
        overall_stats = {}

        for metric in metrics_names:
            values = [r["metrics"][metric] for r in all_results]
            overall_stats[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }

        # Save detailed results to JSON
        results_dict = {
            "config": vars(args),
            "overall_statistics": overall_stats,
            "individual_results": all_results,
            "failed_samples": failed_samples,
            "summary": {
                "total_samples": len(test_files),
                "successful_evaluations": len(all_results),
                "failed_evaluations": len(failed_samples),
                "success_rate": len(all_results) / len(test_files) * 100
                if len(test_files) > 0
                else 0,
            },
        }

        results_file = output_dir / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        # Print comprehensive summary
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE - SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(test_files)}")
        print(f"Successful: {len(all_results)}")
        print(f"Failed: {len(failed_samples)}")
        print(
            f"Success rate: {len(all_results) / len(test_files) * 100:.1f}%"
            if len(test_files) > 0
            else "N/A"
        )

        if len(all_results) > 0:
            print("\nOVERALL METRICS:")
            print(
                f"PSNR:  {overall_stats['psnr']['mean']:.2f} ± {overall_stats['psnr']['std']:.2f} dB"
            )
            print(
                f"       [Min: {overall_stats['psnr']['min']:.2f}, Max: {overall_stats['psnr']['max']:.2f}]"
            )
            print(
                f"SSIM:  {overall_stats['ssim']['mean']:.4f} ± {overall_stats['ssim']['std']:.4f}"
            )
            print(
                f"       [Min: {overall_stats['ssim']['min']:.4f}, Max: {overall_stats['ssim']['max']:.4f}]"
            )
            print(
                f"L1:    {overall_stats['l1']['mean']:.2f} ± {overall_stats['l1']['std']:.2f}"
            )
            print(
                f"L2:    {overall_stats['l2']['mean']:.2f} ± {overall_stats['l2']['std']:.2f}"
            )

            # Find best and worst samples
            best_psnr = max(all_results, key=lambda x: x["metrics"]["psnr"])
            worst_psnr = min(all_results, key=lambda x: x["metrics"]["psnr"])
            best_ssim = max(all_results, key=lambda x: x["metrics"]["ssim"])
            worst_ssim = min(all_results, key=lambda x: x["metrics"]["ssim"])

            print("\nBEST/WORST SAMPLES:")
            print(
                f"Best PSNR:  {best_psnr['filename']} - {best_psnr['metrics']['psnr']:.2f} dB"
            )
            print(
                f"Worst PSNR: {worst_psnr['filename']} - {worst_psnr['metrics']['psnr']:.2f} dB"
            )
            print(
                f"Best SSIM:  {best_ssim['filename']} - {best_ssim['metrics']['ssim']:.4f}"
            )
            print(
                f"Worst SSIM: {worst_ssim['filename']} - {worst_ssim['metrics']['ssim']:.4f}"
            )

        print(f"\nResults saved to: {output_dir}")
        print(f"Detailed JSON: {results_file}")

        # Save a simple text summary
        summary_file = output_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write("EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"LoRA: {args.lora_path}\n")
            f.write(f"Test data: {args.test_data_dir}\n")
            f.write(f"Resolution: {args.resolution}\n")
            f.write(f"Total samples: {len(test_files)}\n")
            f.write(f"Successful evaluations: {len(all_results)}\n")
            if len(all_results) > 0:
                f.write(
                    f"Success rate: {len(all_results) / len(test_files) * 100:.1f}%\n\n"
                )
                f.write("OVERALL METRICS:\n")
                f.write(
                    f"PSNR:  {overall_stats['psnr']['mean']:.2f} ± {overall_stats['psnr']['std']:.2f} dB\n"
                )
                f.write(
                    f"SSIM:  {overall_stats['ssim']['mean']:.4f} ± {overall_stats['ssim']['std']:.4f}\n"
                )
                f.write(
                    f"L1:    {overall_stats['l1']['mean']:.2f} ± {overall_stats['l1']['std']:.2f}\n"
                )
                f.write(
                    f"L2:    {overall_stats['l2']['mean']:.2f} ± {overall_stats['l2']['std']:.2f}\n\n"
                )
                f.write("BEST/WORST SAMPLES:\n")
                f.write(
                    f"Best PSNR:  {best_psnr['filename']} - {best_psnr['metrics']['psnr']:.2f} dB\n"
                )
                f.write(
                    f"Worst PSNR: {worst_psnr['filename']} - {worst_psnr['metrics']['psnr']:.2f} dB\n"
                )
                f.write(
                    f"Best SSIM:  {best_ssim['filename']} - {best_ssim['metrics']['ssim']:.4f}\n"
                )
                f.write(
                    f"Worst SSIM: {worst_ssim['filename']} - {worst_ssim['metrics']['ssim']:.4f}\n"
                )

    # Print failed samples if any
    if failed_samples:
        print("\nFAILED SAMPLES:")
        for failed in failed_samples[:10]:  # Show first 10 failures
            print(f"  {failed['filename']}: {failed['reason']}")
        if len(failed_samples) > 10:
            print(f"  ... and {len(failed_samples) - 10} more")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LoRA model on ALL samples")
    parser.add_argument(
        "--model_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Base model path",
    )
    parser.add_argument(
        "--lora_path", type=str, required=True, help="Path to LoRA weights directory"
    )
    parser.add_argument(
        "--test_data_dir", type=str, required=True, help="Path to test data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="full_evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,  # CHANGED: Default to 64 since you trained on 64x64
        help="Image resolution (should match training - you trained on 64x64!)",
    )
    parser.add_argument(
        "--num_steps", type=int, default=20, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5, help="Guidance scale"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Save individual images in addition to composites",
    )

    args = parser.parse_args()

    # IMPORTANT WARNING
    if args.resolution != 64:
        print(
            f"\n⚠️  WARNING: You trained on 64x64 but are evaluating on {args.resolution}x{args.resolution}"
        )
        print(
            "   This may cause poor results. Use --resolution 64 for proper evaluation!"
        )
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != "y":
            print("Evaluation cancelled.")
            exit()

    evaluate_all_samples(args)
