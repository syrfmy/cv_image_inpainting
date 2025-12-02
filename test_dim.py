#!/usr/bin/env python3
"""
Test UNet dimensions
"""

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel


def test_unet_dimensions():
    print("Testing UNet dimensions...")

    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
    )

    print(f"UNet config:")
    print(f"  Sample size: {unet.config.sample_size}")
    print(f"  In channels: {unet.config.in_channels}")
    print(f"  Out channels: {unet.config.out_channels}")
    print(f"  Block out channels: {unet.config.block_out_channels}")
    print(f"  Cross attention dim: {unet.config.cross_attention_dim}")

    # Test forward pass
    batch_size = 2

    # Test with 64x64 latents (for 512x512 images)
    print(f"\nTest 1: 64x64 latents (for 512x512 images):")
    latents_64 = torch.randn(batch_size, 4, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,)).long()
    encoder_hidden_states = torch.randn(batch_size, 77, 768)

    try:
        output = unet(latents_64, timesteps, encoder_hidden_states).sample
        print(f"  Input shape: {latents_64.shape}")
        print(f"  Output shape: {output.shape}")
        print("  ✓ Success!")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Test with 8x8 latents (for 64x64 images)
    print(f"\nTest 2: 8x8 latents (for 64x64 images):")
    latents_8 = torch.randn(batch_size, 4, 8, 8)

    try:
        output = unet(latents_8, timesteps, encoder_hidden_states).sample
        print(f"  Input shape: {latents_8.shape}")
        print(f"  Output shape: {output.shape}")
        print("  ✓ Success!")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Test VAE dimensions
    print(f"\nTesting VAE dimensions:")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae"
    )

    # Test encoding 512x512 images
    image_512 = torch.randn(batch_size, 3, 512, 512)
    with torch.no_grad():
        latents = vae.encode(image_512).latent_dist.sample()
        print(f"  512x512 image -> latents shape: {latents.shape}")

    # Test encoding 64x64 images
    image_64 = torch.randn(batch_size, 3, 64, 64)
    with torch.no_grad():
        latents = vae.encode(image_64).latent_dist.sample()
        print(f"  64x64 image -> latents shape: {latents.shape}")


if __name__ == "__main__":
    test_unet_dimensions()
