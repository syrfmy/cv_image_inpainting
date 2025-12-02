#!/usr/bin/env python3
"""
Evaluate Stable Diffusion with LoRA - Clean Version
"""

import torch
from diffusers import StableDiffusionPipeline
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate Stable Diffusion with LoRA")
    
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="Base model name")
    parser.add_argument("--lora_path", type=str, required=True,
                       help="Path to LoRA weights")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory")
    parser.add_argument("--prompts", nargs="+", 
                       default=["emoji 1", "emoji 2", "emoji 3", "emoji 4"],
                       help="Prompts for generation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load pipeline
    print("Loading model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None
    )
    
    # Load LoRA weights
    print(f"Loading LoRA weights from {args.lora_path}")
    if args.lora_path.endswith('.safetensors'):
        pipe.unet.load_attn_procs(args.lora_path)
    else:
        # Try to load as state dict
        pipe.load_lora_weights(args.lora_path)
    
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