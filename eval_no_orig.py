#!/usr/bin/env python3
"""
Complete Evaluation Script for LoRA Model
Clean version without text on composites
Works WITHOUT original images - only evaluates visual quality
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


def calculate_visual_metrics(generated_img):
    """Calculate visual quality metrics (no reference needed)"""
    img_array = np.array(generated_img).astype(float)

    metrics = {}

    # 1. Brightness (mean pixel value)
    metrics["brightness"] = np.mean(img_array) / 255.0

    # 2. Contrast (standard deviation)
    metrics["contrast"] = np.std(img_array) / 255.0

    # 3. Color saturation (variance in color channels)
    if len(img_array.shape) == 3:
        channel_std = [np.std(img_array[:, :, i]) for i in range(3)]
        metrics["saturation"] = np.mean(channel_std) / 255.0
    else:
        metrics["saturation"] = 0.0

    # 4. Edge sharpness (Laplacian variance)
    from scipy.ndimage import laplacian

    gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
    laplacian_var = np.var(laplacian(gray))
    metrics["sharpness"] = min(laplacian_var / 1000.0, 1.0)  # Normalize

    # 5. Overall quality score (weighted combination)
    metrics["quality_score"] = (
        0.3 * metrics["brightness"]
        + 0.3 * metrics["contrast"]
        + 0.2 * metrics["saturation"]
        + 0.2 * metrics["sharpness"]
    )

    return metrics


def create_clean_composite(erased_img, mask_img, generated_img, target_size=(64, 64)):
    """Create a clean 2x2 composite without text"""
    # Create 2x2 grid
    composite = Image.new("RGB", (target_size[0] * 2, target_size[1] * 2))

    # Top row: Erased and Mask
    composite.paste(erased_img.resize(target_size, Image.LANCZOS), (0, 0))
    composite.paste(
        mask_img.convert("RGB").resize(target_size, Image.NEAREST), (target_size[0], 0)
    )

    # Bottom row: Blank (original placeholder) and Generated
    blank_img = Image.new("RGB", target_size, color=(240, 240, 240))
    composite.paste(blank_img, (0, target_size[1]))
    composite.paste(
        generated_img.resize(target_size, Image.LANCZOS),
        (target_size[0], target_size[1]),
    )

    return composite


def evaluate_all_samples(args):
    """Evaluate model on ALL test samples (NO original images needed)"""
    print("=" * 60)
    print("EVALUATION - VISUAL QUALITY ONLY (No Original Images Needed)")
    print("=" * 60)

    TARGET_SIZE = (64, 64)
    print(f"Resolution: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print("Note: Evaluating visual quality only (no PSNR/L1/L2 comparisons)")

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
    lora_loaded = False
    try:
        if hasattr(pipe, "load_lora_weights"):
            pipe.load_lora_weights(args.lora_path)
            lora_loaded = True
        else:
            pipe.unet.load_attn_procs(args.lora_path)
            lora_loaded = True
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        # Try safetensors
        try:
            import safetensors.torch

            safetensors_path = os.path.join(
                args.lora_path, "pytorch_lora_weights.safetensors"
            )
            if os.path.exists(safetensors_path):
                state_dict = safetensors.torch.load_file(safetensors_path, device="cpu")
                pipe.unet.load_state_dict(state_dict, strict=False)
                lora_loaded = True
            else:
                print("No safetensors file found")
        except Exception as e:
            print(f"Error loading safetensors: {e}")

    if not lora_loaded:
        print("WARNING: LoRA weights not loaded! Using base model only.")

    pipe = pipe.to(device)

    # Find test files
    test_data_path = Path(args.test_data_dir)
    erased_dir = test_data_path / "erased"
    masks_dir = test_data_path / "masks"

    if not erased_dir.exists():
        print(f"ERROR: Erased directory not found: {erased_dir}")
        return

    if not masks_dir.exists():
        print(f"WARNING: Masks directory not found: {masks_dir}")
        print("Will try to use empty masks for all images")
        use_masks = False
    else:
        use_masks = True

    test_files = list(erased_dir.glob("*_erased.png"))
    if not test_files:
        test_files = list(erased_dir.glob("*.png"))
    if not test_files:
        test_files = list(erased_dir.glob("*.jpg"))
    if not test_files:
        test_files = list(erased_dir.glob("*.jpeg"))

    if not test_files:
        print("No test files found")
        return

    print(f"Found {len(test_files)} test samples")
    print(f"Using masks: {use_masks}")

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

        # Find mask if available
        mask_file = None
        if use_masks:
            mask_patterns = [
                masks_dir / f"{base_name}_mask.png",
                masks_dir / f"{base_name}.png",
                masks_dir / f"{base_name}_mask.jpg",
                masks_dir / f"{base_name}.jpg",
            ]

            for pattern in mask_patterns:
                if pattern.exists():
                    mask_file = pattern
                    break

        try:
            # Load erased image
            erased_img = Image.open(test_file).convert("RGB")

            # Load or create mask
            if mask_file and mask_file.exists():
                mask_img = Image.open(mask_file).convert("L")
            else:
                # Create a blank mask (inpaint entire image)
                mask_img = Image.new("L", erased_img.size, color=0)

        except Exception as e:
            failed_samples.append({"filename": filename, "reason": f"Load error: {e}"})
            continue

        # Resize to target size
        erased_img = erased_img.resize(TARGET_SIZE, Image.LANCZOS)
        mask_img = mask_img.resize(TARGET_SIZE, Image.NEAREST)

        # Invert mask (for inpainting: 1=damage area, 0=intact)
        mask_array = np.array(mask_img)
        mask_inverted = Image.fromarray(255 - mask_array).convert("L")

        # Generate
        prompt = "a damaged picture of a single emoji that needs to be repaired"

        # Optional: Custom prompt per file
        if args.prompt_file and os.path.exists(args.prompt_file):
            try:
                with open(args.prompt_file, "r") as f:
                    prompts = json.load(f)
                if base_name in prompts:
                    prompt = prompts[base_name]
            except:
                pass

        try:
            generated_img = pipe(
                prompt=prompt,
                image=erased_img,
                mask_image=mask_inverted,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                height=TARGET_SIZE[1],
                width=TARGET_SIZE[0],
                generator=torch.Generator(device=device).manual_seed(args.seed)
                if args.seed
                else None,
            ).images[0]
        except Exception as e:
            failed_samples.append(
                {"filename": filename, "reason": f"Generation error: {e}"}
            )
            continue

        # Resize if needed
        if generated_img.size != TARGET_SIZE:
            generated_img = generated_img.resize(TARGET_SIZE, Image.LANCZOS)

        # Calculate visual quality metrics
        metrics = calculate_visual_metrics(generated_img)

        # Create clean composite
        composite = create_clean_composite(
            erased_img, mask_img, generated_img, TARGET_SIZE
        )

        # Save composite
        composite_path = composites_dir / f"{filename}_composite.png"
        composite.save(composite_path)

        # Save individual if requested
        if args.save_individual:
            generated_img.save(individual_dir / f"{filename}_generated.png")
            erased_img.save(individual_dir / f"{filename}_erased.png")
            mask_img.save(individual_dir / f"{filename}_mask.png")

        # Store result
        all_results.append(
            {
                "filename": filename,
                "quality_score": float(metrics["quality_score"]),
                "brightness": float(metrics["brightness"]),
                "contrast": float(metrics["contrast"]),
                "saturation": float(metrics["saturation"]),
                "sharpness": float(metrics["sharpness"]),
                "composite_path": str(composite_path.relative_to(output_dir)),
                "prompt_used": prompt,
            }
        )

    # Generate summary
    if all_results:
        # Calculate statistics
        quality_scores = [r["quality_score"] for r in all_results]
        brightness_values = [r["brightness"] for r in all_results]
        contrast_values = [r["contrast"] for r in all_results]

        stats = {
            "quality_mean": float(np.mean(quality_scores)),
            "quality_std": float(np.std(quality_scores)),
            "quality_min": float(np.min(quality_scores)),
            "quality_max": float(np.max(quality_scores)),
            "brightness_mean": float(np.mean(brightness_values)),
            "contrast_mean": float(np.mean(contrast_values)),
            "saturation_mean": float(np.mean([r["saturation"] for r in all_results])),
            "sharpness_mean": float(np.mean([r["sharpness"] for r in all_results])),
            "total_samples": len(all_results),
            "failed_samples": len(failed_samples),
        }

        # Find best and worst by quality score
        best_sample = max(all_results, key=lambda x: x["quality_score"])
        worst_sample = min(all_results, key=lambda x: x["quality_score"])

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
        print(f"\n{'=' * 60}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"SUCCESS: {len(all_results)}/{len(test_files)} samples")
        print(f"Average Quality Score: {stats['quality_mean']:.3f}")
        print(
            f"Best Quality: {best_sample['quality_score']:.3f} ({best_sample['filename']})"
        )
        print(
            f"Worst Quality: {worst_sample['quality_score']:.3f} ({worst_sample['filename']})"
        )
        print(f"\nVisual Metrics:")
        print(f"  Brightness: {stats['brightness_mean']:.3f}")
        print(f"  Contrast: {stats['contrast_mean']:.3f}")
        print(f"  Saturation: {stats['saturation_mean']:.3f}")
        print(f"  Sharpness: {stats['sharpness_mean']:.3f}")
        print(f"\nOutputs:")
        print(f"  Composites saved to: {composites_dir}")
        print(f"  Full results: {results_file}")
        print(f"{'=' * 60}")

    if failed_samples:
        print(f"\nFailed samples: {len(failed_samples)}")
        if len(failed_samples) <= 10:
            for failed in failed_samples:
                print(f"  {failed['filename']}: {failed['reason']}")
        else:
            print(f"  (Showing first 10 of {len(failed_samples)})")
            for failed in failed_samples[:10]:
                print(f"  {failed['filename']}: {failed['reason']}")

    print("\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA model (NO original images needed)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Path to base model",
    )
    parser.add_argument(
        "--lora_path", type=str, required=True, help="Path to LoRA weights directory"
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=True,
        help="Directory with test images (erased/ and masks/ folders)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
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
        "--save_individual", action="store_true", help="Save individual output images"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        help="Optional JSON file with custom prompts for each image",
    )

    args = parser.parse_args()

    print(f"Evaluating with LoRA: {args.lora_path}")
    print(f"Test data: {args.test_data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Note: No original images needed - evaluating visual quality only")

    evaluate_all_samples(args)
