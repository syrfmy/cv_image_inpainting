#!/usr/bin/env python3
"""
Simple script to check LoRA weights and test loading
"""

import os
from pathlib import Path

import torch


def check_lora_weights(lora_path):
    """Check what's in the LoRA directory"""
    lora_dir = Path(lora_path)

    print(f"Checking LoRA directory: {lora_dir}")
    print(f"Directory exists: {lora_dir.exists()}")

    if not lora_dir.exists():
        print("ERROR: LoRA directory does not exist!")
        return

    # List all files
    print("\nFiles in directory:")
    for file in lora_dir.iterdir():
        print(f"  - {file.name} ({file.stat().st_size / 1024:.1f} KB)")

    # Check for common LoRA files
    common_files = [
        "pytorch_lora_weights.safetensors",
        "pytorch_lora_weights.bin",
        "lora_weights.safetensors",
        "lora_weights.bin",
        "pytorch_model.bin",
    ]

    print("\nLooking for LoRA weight files:")
    found = False
    for file_name in common_files:
        file_path = lora_dir / file_name
        if file_path.exists():
            print(f"  ✓ Found: {file_name}")
            found = True

    if not found:
        print("  ✗ No common LoRA files found")

    # Check for any safetensors or bin files
    print("\nAll safetensors files:")
    for file in lora_dir.glob("*.safetensors"):
        print(f"  - {file.name}")

    print("\nAll bin files:")
    for file in lora_dir.glob("*.bin"):
        print(f"  - {file.name}")

    # Try to load as safetensors to check
    print("\nTrying to load safetensors file...")
    safetensors_files = list(lora_dir.glob("*.safetensors"))
    if safetensors_files:
        try:
            import safetensors.torch

            weights = safetensors.torch.load_file(safetensors_files[0])
            print(f"  ✓ Successfully loaded {safetensors_files[0].name}")
            print(f"  Number of weight tensors: {len(weights)}")
            for key in list(weights.keys())[:5]:  # Show first 5 keys
                print(f"    {key}: {weights[key].shape}")
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True)
    args = parser.parse_args()

    check_lora_weights(args.lora_path)
