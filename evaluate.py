#!/usr/bin/env python3
"""
Complete Evaluation Script for LoRA Model
Optimized for 64x64 images with proper collages
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
    """Calculate metrics between two images"""
    # Ensure both images are the same size
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.LANCZOS)

    # Convert to numpy arrays
    arr1 = np.array(img1).astype(float)
    arr2 = np.array(img2).astype(float)

    metrics = {}

    # PSNR
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        metrics["psnr"] = float("inf")
    else:
        max_pixel = 255.0
        metrics["psnr"] = 20 * np.log10(max_pixel / np.sqrt(mse))

    # L1/L2 distances
    metrics["l1"] = np.mean(np.abs(arr1 - arr2))
    metrics["l2"] = np.sqrt(mse)

    return metrics


def create_collage(
    erased_img,
    mask_img,
    orig_img,
    generated_img,
    filename,
    metrics,
    save_dir,
    target_size=(64, 64),
):
    """Create a comprehensive collage image with all components and metrics"""

    # Create a larger canvas to hold everything
    # Layout: 2x2 grid of images with space for text
    cell_width = target_size[0]
    cell_height = target_size[1]
    padding = 10
    text_height = 40

    # Calculate total canvas size
    canvas_width = cell_width * 2 + padding * 3
    canvas_height = cell_height * 2 + padding * 3 + text_height * 2

    # Create canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
    draw = ImageDraw.Draw(canvas)

    # Try to load font
    try:
        font_large = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title
    draw.text(
        (canvas_width // 2, 5),
        f"Evaluation: {filename}",
        fill="black",
        font=font_large,
        anchor="mt",
    )

    # Row 1: Erased and Mask
    y_offset = text_height

    # Erased image
    canvas.paste(erased_img.resize(target_size, Image.LANCZOS), (padding, y_offset))
    draw.text(
        (padding + cell_width // 2, y_offset + cell_height + 5),
        "Erased (Input)",
        fill="black",
        font=font_small,
        anchor="mt",
    )

    # Mask image
    canvas.paste(
        mask_img.convert("RGB").resize(target_size, Image.NEAREST),
        (cell_width + padding * 2, y_offset),
    )
    draw.text(
        (cell_width + padding * 2 + cell_width // 2, y_offset + cell_height + 5),
        "Mask",
        fill="black",
        font=font_small,
        anchor="mt",
    )

    # Row 2: Original and Generated
    y_offset += cell_height + text_height

    # Original image
    canvas.paste(orig_img.resize(target_size, Image.LANCZOS), (padding, y_offset))
    draw.text(
        (padding + cell_width // 2, y_offset + cell_height + 5),
        "Original (Ground Truth)",
        fill="black",
        font=font_small,
        anchor="mt",
    )

    # Generated image
    canvas.paste(
        generated_img.resize(target_size, Image.LANCZOS),
        (cell_width + padding * 2, y_offset),
    )
    draw.text(
        (cell_width + padding * 2 + cell_width // 2, y_offset + cell_height + 5),
        "Generated (Output)",
        fill="black",
        font=font_small,
        anchor="mt",
    )

    # Metrics at the bottom
    y_offset += cell_height + text_height

    metrics_text = f"PSNR: {metrics['psnr']:.2f} dB | L1: {metrics['l1']:.3f} | L2: {metrics['l2']:.3f}"
    draw.text(
        (canvas_width // 2, canvas_height - 25),
        metrics_text,
        fill="black",
        font=font_small,
        anchor="mt",
    )

    # Save
    output_path = save_dir / f"{filename}_collage.png"
    canvas.save(output_path)

    return output_path


def evaluate_all_samples(args):
    """Evaluate model on ALL test samples"""
    print("=" * 60)
    print("COMPLETE EVALUATION - 64x64 IMAGES")
    print("=" * 60)

    # Fixed for 64x64 training
    TARGET_SIZE = (64, 64)
    print(f"Target resolution: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("Loading pipeline...")
    # Load pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Load LoRA weights
    print(f"Loading LoRA weights from {args.lora_path}")
    try:
        if hasattr(pipe, "load_lora_weights"):
            pipe.load_lora_weights(args.lora_path)
        else:
            pipe.unet.load_attn_procs(args.lora_path)
    except Exception as e:
        print(f"Error loading LoRA weights: {e}")
        # Try alternative loading
        try:
            import safetensors.torch

            safetensors_path = os.path.join(
                args.lora_path, "pytorch_lora_weights.safetensors"
            )
            if os.path.exists(safetensors_path):
                state_dict = safetensors.torch.load_file(safetensors_path, device="cpu")
                pipe.unet.load_state_dict(state_dict, strict=False)
                print("Loaded from safetensors")
        except:
            print("Warning: Using base model only")

    # Move to device
    pipe = pipe.to(device)

    # Find test images
    test_data_path = Path(args.test_data_dir)
    erased_dir = test_data_path / "erased"
    masks_dir = test_data_path / "masks"
    orig_dir = test_data_path / "orig"

    if not all([erased_dir.exists(), masks_dir.exists(), orig_dir.exists()]):
        print("Error: Missing required directories")
        return

    # Get all erased files
    test_files = list(erased_dir.glob("*_erased.png"))
    if not test_files:
        test_files = list(erased_dir.glob("*.png"))

    if not test_files:
        print(f"No test files found in {erased_dir}")
        return

    print(f"Found {len(test_files)} samples to evaluate")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    collages_dir = output_dir / "collages"
    collages_dir.mkdir(exist_ok=True)

    if args.save_individual:
        individual_dir = output_dir / "individual"
        individual_dir.mkdir(exist_ok=True)

    # Store results
    all_results = []
    failed_samples = []

    # Progress bar
    progress_bar = tqdm(test_files, desc="Evaluating", unit="sample")

    for test_file in progress_bar:
        filename = test_file.stem

        # Extract base name
        if "_erased" in filename:
            base_name = filename.replace("_erased", "")
        else:
            base_name = filename

        # Find corresponding files
        mask_file = masks_dir / f"{base_name}_mask.png"
        orig_file = orig_dir / f"{base_name}.png"

        if not mask_file.exists() or not orig_file.exists():
            failed_samples.append({"filename": filename, "reason": "Missing files"})
            continue

        try:
            # Load images
            erased_img = Image.open(test_file).convert("RGB")
            mask_img = Image.open(mask_file).convert("L")
            orig_img = Image.open(orig_file).convert("RGB")
        except Exception as e:
            failed_samples.append({"filename": filename, "reason": f"Load error: {e}"})
            continue

        # Resize to 64x64
        erased_img = erased_img.resize(TARGET_SIZE, Image.LANCZOS)
        mask_img = mask_img.resize(TARGET_SIZE, Image.NEAREST)
        orig_img_resized = orig_img.resize(TARGET_SIZE, Image.LANCZOS)

        # Invert mask for Stable Diffusion (black damage -> white damage)
        mask_array = np.array(mask_img)
        mask_inverted = Image.fromarray(255 - mask_array).convert("L")

        # Generate
        prompt = "a damaged picture of a single emoji that needs to be repaired"

        try:
            generated_img = pipe(
                prompt=prompt,
                image=erased_img,
                mask_image=mask_inverted,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                height=TARGET_SIZE[1],
                width=TARGET_SIZE[0],
            ).images[0]
        except Exception as e:
            failed_samples.append(
                {"filename": filename, "reason": f"Generation error: {e}"}
            )
            continue

        # Resize if needed
        if generated_img.size != TARGET_SIZE:
            generated_img = generated_img.resize(TARGET_SIZE, Image.LANCZOS)

        # Calculate metrics
        metrics = calculate_metrics(generated_img, orig_img_resized)

        # Create comprehensive collage
        collage_path = create_collage(
            erased_img,
            mask_img,
            orig_img_resized,
            generated_img,
            filename,
            metrics,
            collages_dir,
            TARGET_SIZE,
        )

        # Save individual images if requested
        if args.save_individual:
            erased_img.save(individual_dir / f"{filename}_erased.png")
            mask_img.save(individual_dir / f"{filename}_mask.png")
            mask_inverted.save(individual_dir / f"{filename}_mask_inverted.png")
            orig_img_resized.save(individual_dir / f"{filename}_original.png")
            generated_img.save(individual_dir / f"{filename}_generated.png")

        # Store result
        sample_result = {
            "filename": filename,
            "metrics": metrics,
            "collage_path": str(collage_path),
            "psnr": metrics["psnr"],
            "l1": metrics["l1"],
            "l2": metrics["l2"],
        }
        all_results.append(sample_result)

        # Update progress
        progress_bar.set_postfix(
            {"PSNR": f"{metrics['psnr']:.1f}", "L1": f"{metrics['l1']:.3f}"}
        )

    # Generate summary
    if all_results:
        # Calculate statistics
        psnr_values = [r["psnr"] for r in all_results]
        l1_values = [r["l1"] for r in all_results]
        l2_values = [r["l2"] for r in all_results]

        stats = {
            "psnr": {
                "mean": float(np.mean(psnr_values)),
                "std": float(np.std(psnr_values)),
                "min": float(np.min(psnr_values)),
                "max": float(np.max(psnr_values)),
                "median": float(np.median(psnr_values)),
            },
            "l1": {
                "mean": float(np.mean(l1_values)),
                "std": float(np.std(l1_values)),
                "min": float(np.min(l1_values)),
                "max": float(np.max(l1_values)),
                "median": float(np.median(l1_values)),
            },
            "l2": {
                "mean": float(np.mean(l2_values)),
                "std": float(np.std(l2_values)),
                "min": float(np.min(l2_values)),
                "max": float(np.max(l2_values)),
                "median": float(np.median(l2_values)),
            },
        }

        # Find best and worst
        best_psnr = max(all_results, key=lambda x: x["psnr"])
        worst_psnr = min(all_results, key=lambda x: x["psnr"])
        best_l1 = min(all_results, key=lambda x: x["l1"])
        worst_l1 = max(all_results, key=lambda x: x["l1"])

        # Save results
        results_dict = {
            "config": vars(args),
            "statistics": stats,
            "samples": all_results,
            "failed": failed_samples,
            "summary": {
                "total": len(test_files),
                "successful": len(all_results),
                "failed": len(failed_samples),
                "success_rate": f"{(len(all_results) / len(test_files) * 100):.1f}%",
            },
        }

        results_file = output_dir / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(test_files)}")
        print(f"Successful: {len(all_results)}")
        print(f"Failed: {len(failed_samples)}")
        print(f"Success rate: {(len(all_results) / len(test_files) * 100):.1f}%")

        print(f"\nPSNR: {stats['psnr']['mean']:.2f} ± {stats['psnr']['std']:.2f} dB")
        print(f"     Best: {best_psnr['psnr']:.2f} dB ({best_psnr['filename']})")
        print(f"     Worst: {worst_psnr['psnr']:.2f} dB ({worst_psnr['filename']})")

        print(f"\nL1 Distance: {stats['l1']['mean']:.4f} ± {stats['l1']['std']:.4f}")
        print(f"     Best: {best_l1['l1']:.4f} ({best_l1['filename']})")
        print(f"     Worst: {worst_l1['l1']:.4f} ({worst_l1['filename']})")

        print(f"\nResults saved to: {output_dir}")
        print(f"Collages: {collages_dir}")
        print(f"JSON report: {results_file}")

        # Create quick visual summary
        if len(all_results) >= 9:
            print("\nCreating visual summary grid...")
            create_summary_grid(all_results, collages_dir, output_dir)

    # Show failures
    if failed_samples:
        print(f"\nFailed samples ({len(failed_samples)}):")
        for failed in failed_samples[:5]:
            print(f"  {failed['filename']}: {failed['reason']}")
        if len(failed_samples) > 5:
            print(f"  ... and {len(failed_samples) - 5} more")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


def create_summary_grid(all_results, collages_dir, output_dir):
    """Create a grid of sample collages for quick visual inspection"""
    # Sort by PSNR (best first)
    sorted_results = sorted(all_results, key=lambda x: x["psnr"], reverse=True)

    # Take top 9 results
    top_results = sorted_results[:9]

    # Create 3x3 grid
    grid_size = 3
    collage_size = 64 * 2 + 20  # Approximate collage size

    grid_width = collage_size * grid_size
    grid_height = collage_size * grid_size

    grid_canvas = Image.new("RGB", (grid_width, grid_height), color="white")

    for i, result in enumerate(top_results):
        row = i // grid_size
        col = i % grid_size

        try:
            collage = Image.open(result["collage_path"])
            # Resize collage to fit grid
            collage_resized = collage.resize(
                (collage_size, collage_size), Image.LANCZOS
            )

            x_pos = col * collage_size
            y_pos = row * collage_size

            grid_canvas.paste(collage_resized, (x_pos, y_pos))
        except:
            continue

    # Add title
    draw = ImageDraw.Draw(grid_canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    draw.text(
        (grid_width // 2, 10),
        "Top 9 Results (by PSNR)",
        fill="black",
        font=font,
        anchor="mt",
    )

    grid_path = output_dir / "summary_grid.png"
    grid_canvas.save(grid_path)
    print(f"Summary grid saved to: {grid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LoRA model on 64x64 images")
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
        default="evaluation_results_64x64",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_steps", type=int, default=20, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5, help="Guidance scale"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_individual", action="store_true", help="Save individual images"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )

    args = parser.parse_args()

    # Override resolution to 64
    print(f"\nRunning evaluation at 64x64 resolution (matching training)")
    print(f"LoRA: {args.lora_path}")
    print(f"Test data: {args.test_data_dir}")
    print(f"Output: {args.output_dir}")

    evaluate_all_samples(args)
