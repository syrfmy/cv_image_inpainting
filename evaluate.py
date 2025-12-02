#!/usr/bin/env python3
"""
Simple Evaluation Script
"""

import argparse
import os

import torch
from diffusers import StableDiffusionPipeline


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stable Diffusion")

    parser.add_argument(
        "--model_name",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base model name",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA weights (.safetensors)",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to checkpoint (.pt)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Output directory"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["emoji 1", "emoji 2", "emoji 3", "emoji 4"],
        help="Prompts for generation",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pipeline
    print("Loading model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    )

    # Load weights if provided
    if args.lora_path:
        print(f"Loading LoRA weights from {args.lora_path}")
        pipe.unet.load_attn_procs(args.lora_path)
    elif args.checkpoint_path:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict:
            pipe.unet.load_state_dict(state_dict["state_dict"], strict=False)
        else:
            pipe.unet.load_state_dict(state_dict, strict=False)

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    # Generate images
    print(f"Generating {len(args.prompts)} images...")
    for i, prompt in enumerate(args.prompts):
        print(f"  Generating: '{prompt}'")
        image = pipe(prompt, num_inference_steps=50).images[0]

        # Save image
        filename = f"generated_{i:03d}.png"
        save_path = os.path.join(args.output_dir, filename)
        image.save(save_path)
        print(f"    Saved to: {save_path}")

    print(f"\nAll images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
