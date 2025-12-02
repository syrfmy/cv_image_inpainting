#!/usr/bin/env python3
"""
Simple Evaluation Script
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stable Diffusion LoRA")

    parser.add_argument(
        "--model_name",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base model name",
    )
    parser.add_argument(
        "--lora_path", type=str, required=True, help="Path to LoRA weights"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--prompt", type=str, default="a photo of a cat", help="Prompt for generation"
    )
    parser.add_argument(
        "--num_images", type=int, default=4, help="Number of images to generate"
    )

    args = parser.parse_args()

    # Load pipeline with LoRA
    print("Loading model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # Load LoRA weights
    pipe.load_lora_weights(args.lora_path)

    # Use DDIM scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate images
    print(f"Generating {args.num_images} images...")
    for i in tqdm(range(args.num_images)):
        image = pipe(args.prompt, num_inference_steps=50).images[0]
        image.save(os.path.join(args.output_dir, f"generated_{i:03d}.png"))

    print(f"Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
