#!/usr/bin/env python3
"""
Quick Test Script for LoRA Model
Runs a quick evaluation on a few samples.
"""

import argparse
import os

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


def quick_test(args):
    """Quick test of the trained model"""

    print("Loading pipeline...")

    # Load base pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # Load LoRA weights
    pipe.unet.load_attn_procs(args.lora_path)

    # Move to device
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    # Find test images
    import glob

    test_files = glob.glob(os.path.join(args.test_data_dir, "erased", "*.png"))
    test_files = test_files[:3]  # Test on first 3 images

    print(f"Testing on {len(test_files)} samples...")

    for test_file in test_files:
        filename = os.path.basename(test_file).replace("_erased.png", "")

        # Load corresponding mask and original
        mask_file = os.path.join(args.test_data_dir, "masks", f"{filename}_mask.png")
        orig_file = os.path.join(args.test_data_dir, "orig", f"{filename}.png")

        if not os.path.exists(mask_file) or not os.path.exists(orig_file):
            print(f"Missing files for {filename}, skipping...")
            continue

        # Load images
        image = Image.open(test_file).convert("RGB")
        mask_image = Image.open(mask_file).convert("L")
        original = Image.open(orig_file).convert("RGB")

        # Resize
        image = image.resize((512, 512))
        mask_image = mask_image.resize((512, 512))
        original = original.resize((512, 512))

        # Generate
        prompt = "a picture of a single emoji"

        print(f"Generating for {filename}...")
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            num_inference_steps=20,
            guidance_scale=7.5,
        ).images[0]

        # Save results
        output_dir = os.path.join(args.lora_path, "quick_test")
        os.makedirs(output_dir, exist_ok=True)

        # Create composite
        composite = Image.new("RGB", (512 * 4, 512))
        composite.paste(image, (0, 0))
        composite.paste(mask_image.convert("RGB"), (512, 0))
        composite.paste(original, (512 * 2, 0))
        composite.paste(result, (512 * 3, 0))

        composite.save(os.path.join(output_dir, f"{filename}_test.png"))
        print(f"Saved result for {filename}")

    print(f"\nQuick test complete! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="runwayml/stable-diffusion-inpainting"
    )
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--test_data_dir", type=str, required=True)

    args = parser.parse_args()
    quick_test(args)
