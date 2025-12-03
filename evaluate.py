#!/usr/bin/env python3
"""
Complete Evaluation Script for LoRA Model
Clean version without text on composites
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from tqdm import tqdm


def calculate_metrics(img1, img2):
    """Calculate metrics between two images"""
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.LANCZOS)

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


def create_clean_composite(
    erased_img, mask_img, orig_img, generated_img, target_size=(64, 64)
):
    """Create a clean 2x2 composite without text"""
    # Create 2x2 grid
    composite = Image.new("RGB", (target_size[0] * 2, target_size[1] * 2))

    # Top row: Erased and Mask
    composite.paste(erased_img.resize(target_size, Image.LANCZOS), (0, 0))
    composite.paste(
        mask_img.convert("RGB").resize(target_size, Image.NEAREST), (target_size[0], 0)
    )

    # Bottom row: Original and Generated
    composite.paste(orig_img.resize(target_size, Image.LANCZOS), (0, target_size[1]))
    composite.paste(
        generated_img.resize(target_size, Image.LANCZOS),
        (target_size[0], target_size[1]),
    )

    return composite


def evaluate_all_samples(args):
    """Evaluate model on ALL test samples"""
    print("=" * 60)
    print("EVALUATION - 64x64 IMAGES")
    print("=" * 60)

    TARGET_SIZE = (64, 64)
    print(f"Resolution: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("Loading pipeline...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Load LoRA
    print(f"Loading LoRA from {args.lora_path}")
    try:
        if hasattr(pipe, "load_lora_weights"):
            pipe.load_lora_weights(args.lora_path)
        else:
            pipe.unet.load_attn_procs(args.lora_path)
    except Exception as e:
        print(f"Error: {e}")
        # Try safetensors
        try:
            import safetensors.torch

            safetensors_path = os.path.join(
                args.lora_path, "pytorch_lora_weights.safetensors"
            )
            if os.path.exists(safetensors_path):
                state_dict = safetensors.torch.load_file(safetensors_path, device="cpu")
                pipe.unet.load_state_dict(state_dict, strict=False)
        except:
            pass

    pipe = pipe.to(device)

    # Find test files
    test_data_path = Path(args.test_data_dir)
    erased_dir = test_data_path / "erased"
    masks_dir = test_data_path / "masks"
    # orig_dir = test_data_path / "orig"

    # if not all([erased_dir.exists(), masks_dir.exists(), orig_dir.exists()]):
    #     print("Missing directories")
    #     return

    if not all([erased_dir.exists(), masks_dir.exists()]):
        print("Missing directories")
        return

    test_files = list(erased_dir.glob("*_erased.png"))
    if not test_files:
        test_files = list(erased_dir.glob("*.png"))

    if not test_files:
        print("No test files found")
        return

    print(f"Found {len(test_files)} samples")

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    composites_dir = output_dir / "composites"
    composites_dir.mkdir(exist_ok=True)

    if args.save_individual:
        individual_dir = output_dir / "individual"
        individual_dir.mkdir(exist_ok=True)

    # Results storage
    all_results = []
    failed_samples = []

    # Progress bar
    for test_file in tqdm(test_files, desc="Evaluating"):
        filename = test_file.stem

        # Get base name
        if "_erased" in filename:
            base_name = filename.replace("_erased", "")
        else:
            base_name = filename

        # Find mask and original
        mask_file = masks_dir / f"{base_name}_mask.png"
        # orig_file = orig_dir / f"{base_name}.png"

        if not mask_file.exists():
            failed_samples.append({"filename": filename, "reason": "Missing files"})
            continue

        try:
            # Load images
            erased_img = Image.open(test_file).convert("RGB")
            mask_img = Image.open(mask_file).convert("L")
            # orig_img = Image.open(orig_file).convert("RGB")
        except Exception as e:
            failed_samples.append({"filename": filename, "reason": f"Load error: {e}"})
            continue

        # Resize to 64x64
        erased_img = erased_img.resize(TARGET_SIZE, Image.LANCZOS)
        mask_img = mask_img.resize(TARGET_SIZE, Image.NEAREST)
        # orig_img_resized = orig_img.resize(TARGET_SIZE, Image.LANCZOS)

        # Invert mask
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

        # # Calculate metrics
        # metrics = calculate_metrics(generated_img, orig_img_resized)

        # Create clean composite (no text)
        composite = create_clean_composite(
            erased_img, mask_img, generated_img, TARGET_SIZE
        )

        # Save composite
        composite_path = composites_dir / f"{filename}_composite.png"
        composite.save(composite_path)

        # Save individual if requested
        if args.save_individual:
            generated_img.save(individual_dir / f"{filename}_generated.png")

        # Store result
        # all_results.append(
        #     {
        #         "filename": filename,
        #         "psnr": float(metrics["psnr"]),
        #         "l1": float(metrics["l1"]),
        #         "l2": float(metrics["l2"]),
        #         "composite_path": str(composite_path),
        #     }
        # )

    # Generate summary
    if all_results:
        # Calculate statistics
        psnr_values = [r["psnr"] for r in all_results]
        l1_values = [r["l1"] for r in all_results]

        stats = {
            "psnr_mean": float(np.mean(psnr_values)),
            "psnr_std": float(np.std(psnr_values)),
            "psnr_min": float(np.min(psnr_values)),
            "psnr_max": float(np.max(psnr_values)),
            "l1_mean": float(np.mean(l1_values)),
            "l1_std": float(np.std(l1_values)),
        }

        # Find best and worst
        best_sample = max(all_results, key=lambda x: x["psnr"])
        worst_sample = min(all_results, key=lambda x: x["psnr"])

        # Save results
        results_dict = {
            "config": vars(args),
            "statistics": stats,
            "samples": all_results,
            "failed": failed_samples,
            "best_sample": best_sample,
            "worst_sample": worst_sample,
        }

        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        # Print summary
        print(f"\nSUCCESS: {len(all_results)}/{len(test_files)}")
        print(f"Average PSNR: {stats['psnr_mean']:.2f} dB")
        print(f"Best PSNR: {best_sample['psnr']:.2f} dB ({best_sample['filename']})")
        print(f"Worst PSNR: {worst_sample['psnr']:.2f} dB ({worst_sample['filename']})")
        print(f"\nComposites saved to: {composites_dir}")
        print(f"Full results: {results_file}")

    if failed_samples:
        print(f"\nFailed: {len(failed_samples)} samples")
        for failed in failed_samples[:3]:
            print(f"  {failed['filename']}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LoRA model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
    )

    args = parser.parse_args()

    print(f"Evaluating with LoRA: {args.lora_path}")
    print(f"Test data: {args.test_data_dir}")
    print(f"Output: {args.output_dir}")

    evaluate_all_samples(args)
