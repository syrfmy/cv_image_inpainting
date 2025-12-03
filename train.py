#!/usr/bin/env python3
"""
Enhanced LoRA Training for Emoji Inpainting
Optimized for both attention and convolutional layers
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

logger = get_logger(__name__)


def prepare_inpainting_inputs(erased_image, mask_image, target_size=None):
    """Prepare inputs for inpainting training with proper mask resizing"""
    # Convert erased image to tensor [-1, 1]
    erased_np = np.array(erased_image.convert("RGB"))
    if target_size:
        # Resize to target if provided
        erased_image = erased_image.resize(target_size, Image.BILINEAR)
        erased_np = np.array(erased_image)

    erased_np = erased_np[None].transpose(0, 3, 1, 2)
    erased_tensor = torch.from_numpy(erased_np).to(dtype=torch.float32) / 127.5 - 1.0

    # Convert mask: INVERT so black=damage (1), white=intact (0)
    mask_np = np.array(mask_image.convert("L"))

    # Ensure mask matches erased image size
    if mask_np.shape[:2] != erased_image.size[::-1]:  # H,W vs W,H
        mask_image = mask_image.resize(erased_image.size, Image.NEAREST)
        mask_np = np.array(mask_image)

    mask_np = mask_np.astype(np.float32) / 255.0
    mask_np = 1.0 - mask_np  # INVERT: black(0)->white(1), white(1)->black(0)
    mask_np = mask_np[None, None]  # Add batch and channel dims
    mask_tensor = torch.from_numpy(mask_np)

    return mask_tensor, erased_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced LoRA Training for Emoji Inpainting"
    )

    # Required arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Path to training dataset (should contain erased/, masks/, orig/ folders).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="emoji-lora-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Training parameters
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=3000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate for LoRA layers.",
    )
    parser.add_argument(
        "--conv_learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for convolutional layers (if unfrozen).",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=200,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )

    # Enhanced LoRA parameters
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,  # Higher rank for better inpainting
        help="Rank of LoRA layers.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="Alpha parameter for LoRA scaling.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout probability for LoRA layers.",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to also apply LoRA to text encoder.",
    )
    parser.add_argument(
        "--train_conv_layers",
        action="store_true",
        help="Whether to also train convolutional layers (better for inpainting).",
    )

    # System parameters
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint every X steps.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="a perfect emoji with clean edges and solid colors",
        help="Prompt for validation generation.",
    )

    args = parser.parse_args()

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    # Env settings for better performance
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ["ACCELERATE_DEBUG_MODE"] = "false"

    return args


class EmojiInpaintingDataset(Dataset):
    """Dataset for emoji inpainting training with robust file handling."""

    def __init__(self, data_root, tokenizer, size=512):
        self.size = size
        self.tokenizer = tokenizer
        self.data_root = Path(data_root)

        # Load paths
        self.erased_dir = self.data_root / "erased"
        self.masks_dir = self.data_root / "masks"
        self.orig_dir = self.data_root / "orig"

        # Verify folders
        assert self.erased_dir.exists(), f"erased folder not found: {self.erased_dir}"
        assert self.masks_dir.exists(), f"masks folder not found: {self.masks_dir}"
        assert self.orig_dir.exists(), f"orig folder not found: {self.orig_dir}"

        # Get all files with flexible naming
        self.erased_files = self._collect_files(
            self.erased_dir, ["*_erased.png", "*.png", "*.jpg"]
        )
        print(f"Found {len(self.erased_files)} training samples")

        # Enhanced prompts for better emoji reconstruction
        self.emoji_prompts = [
            "a perfect emoji with clean edges and solid colors",
            "a simple emoji icon with crisp boundaries",
            "a cartoon emoji with vibrant colors and no artifacts",
            "a pixel-perfect emoji symbol",
            "a clean emoji graphic with smooth edges",
            "a vector-style emoji with bold outlines",
            "a digital emoji with perfect symmetry",
            "a classic emoji with simple shapes and colors",
            "a stylized emoji with clear contours",
            "a modern emoji design with flat colors",
        ]

        # Image transforms with augmentation
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size),
                transforms.RandomHorizontalFlip(p=0.3),  # Small augmentation
            ]
        )

    def _collect_files(self, directory, patterns):
        files = []
        for pattern in patterns:
            files.extend(directory.glob(pattern))
        return sorted(list(set(files)))  # Remove duplicates

    def _find_matching_file(self, base_name, directory, suffixes):
        for suffix in suffixes:
            for ext in [".png", ".jpg", ".jpeg"]:
                path = directory / f"{base_name}{suffix}{ext}"
                if path.exists():
                    return path
                # Also try without extension in base_name
                path = directory / f"{base_name.replace(ext, '')}{suffix}{ext}"
                if path.exists():
                    return path
        return None

    def __len__(self):
        return len(self.erased_files)

    def __getitem__(self, index):
        # Get erased image path
        erased_path = self.erased_files[index]
        filename = erased_path.stem

        # Extract base name (handle various naming conventions)
        base_name = filename
        for suffix in ["_erased", "_masked", "_damaged", "_inpaint"]:
            if suffix in filename:
                base_name = filename.split(suffix)[0]
                break

        # Find mask and original files
        mask_path = self._find_matching_file(
            base_name, self.masks_dir, ["_mask", "_masked", "", "_damage"]
        )

        orig_path = self._find_matching_file(
            base_name, self.orig_dir, ["", "_original", "_gt", "_target"]
        )

        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for {erased_path}")
        if orig_path is None:
            raise FileNotFoundError(f"Original not found for {erased_path}")

        # Random prompt for better generalization
        prompt = random.choice(self.emoji_prompts)

        # Load images
        erased_image = Image.open(erased_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")
        orig_image = Image.open(orig_path).convert("RGB")

        # Apply transforms
        orig_image = self.transform(orig_image)
        erased_image = self.transform(erased_image)

        # Resize mask to match exactly
        mask_image = mask_image.resize(orig_image.size, Image.NEAREST)

        # Tokenize prompt
        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        prompt_ids = tokenized.input_ids[0]

        return {
            "erased_image": erased_image,
            "orig_image": orig_image,
            "mask": mask_image,
            "prompt_ids": prompt_ids,
            "prompt": prompt,
        }


def collate_fn(examples, tokenizer):
    """Collate function for the dataloader"""
    # Tokenization
    input_ids = [example["prompt_ids"] for example in examples]
    input_ids = torch.stack(input_ids)

    # Prepare lists for tensors
    orig_tensors = []
    masks = []
    erased_tensors = []
    prompts = []

    for example in examples:
        # Original image (ground truth)
        orig_pil = example["orig_image"]
        orig_np = np.array(orig_pil.convert("RGB"))
        orig_np = orig_np.transpose(2, 0, 1)  # HWC to CHW
        orig_tensor = torch.from_numpy(orig_np).float() / 127.5 - 1.0
        orig_tensors.append(orig_tensor)

        # Prepare mask and erased image
        mask, erased = prepare_inpainting_inputs(
            example["erased_image"], example["mask"], target_size=orig_pil.size
        )
        masks.append(mask.squeeze(0))
        erased_tensors.append(erased.squeeze(0))
        prompts.append(example["prompt"])

    # Stack tensors
    orig_tensors = torch.stack(orig_tensors)
    masks = torch.stack(masks)
    erased_tensors = torch.stack(erased_tensors)

    return {
        "input_ids": input_ids,
        "orig_images": orig_tensors,
        "masks": masks,
        "erased_images": erased_tensors,
        "prompts": prompts,
    }


def setup_lora_for_inpainting(unet, text_encoder=None, args=None):
    """Setup enhanced LoRA configuration for inpainting"""

    # Define target modules for UNet
    unet_target_modules = [
        # Attention layers
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "add_k_proj",
        "add_v_proj",  # Cross-attention
        "proj_in",
        "proj_out",  # Additional projections
        # Convolutional layers (critical for inpainting)
        "conv_in",
        "conv_out",
        "conv_shortcut",
    ]

    # Add more conv layers if requested
    if args.train_conv_layers:
        unet_target_modules.extend(
            [
                "*.conv",
                "*.conv1",
                "*.conv2",
                "*.conv_shortcut",
                "*resnets*.conv*",
                "*upsamplers*.conv*",
                "*downsamplers*.conv*",
            ]
        )

    unet_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=unet_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    # Apply LoRA to UNet
    unet.add_adapter(unet_config)

    # Optionally apply LoRA to text encoder
    text_encoder_lora_config = None
    if text_encoder is not None and args.train_text_encoder:
        text_encoder_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        text_encoder_lora_config = LoraConfig(
            r=args.lora_rank // 2,  # Lower rank for text encoder
            lora_alpha=args.lora_alpha // 2,
            target_modules=text_encoder_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        text_encoder.add_adapter(text_encoder_lora_config)

    return unet_config, text_encoder_lora_config


def main():
    args = parse_args()

    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir,
        kwargs_handlers=[ProjectConfiguration(automatic_checkpoint_naming=True)],
    )

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    # Freeze base models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Set dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to device and cast
    device = accelerator.device
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    # Setup enhanced LoRA
    print("Setting up enhanced LoRA for inpainting...")
    unet_config, text_encoder_config = setup_lora_for_inpainting(
        unet, text_encoder, args
    )

    # Set train mode
    unet.train()
    if args.train_text_encoder:
        text_encoder.train()

    # Collect trainable parameters
    trainable_params = []
    param_groups = []

    # UNet LoRA parameters
    for name, param in unet.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            trainable_params.append(param)

    # Text encoder LoRA parameters
    if args.train_text_encoder:
        for name, param in text_encoder.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
                trainable_params.append(param)

    # Optionally unfreeze critical conv layers
    if args.train_conv_layers:
        for name, param in unet.named_parameters():
            if any(x in name for x in ["conv_in", "conv_out", "conv_shortcut"]):
                param.requires_grad = True
                trainable_params.append(param)

    total_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total_params:,}")
    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    print(
        f"Text encoder parameters: {sum(p.numel() for p in text_encoder.parameters()):,}"
    )

    # Setup optimizer with different learning rates
    optimizer_params = []

    # LoRA parameters with higher LR
    lora_params = [p for n, p in unet.named_parameters() if "lora" in n.lower()]
    if lora_params:
        optimizer_params.append({"params": lora_params, "lr": args.learning_rate})

    # Conv layer parameters with lower LR
    conv_params = [
        p
        for n, p in unet.named_parameters()
        if any(x in n for x in ["conv_in", "conv_out", "conv_shortcut"])
        and p.requires_grad
    ]
    if conv_params:
        optimizer_params.append({"params": conv_params, "lr": args.conv_learning_rate})

    # Text encoder LoRA parameters
    if args.train_text_encoder:
        text_lora_params = [
            p for n, p in text_encoder.named_parameters() if "lora" in n.lower()
        ]
        if text_lora_params:
            optimizer_params.append(
                {"params": text_lora_params, "lr": args.learning_rate / 2}
            )

    optimizer = torch.optim.AdamW(
        optimizer_params,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # Load dataset
    print("Loading dataset...")
    train_dataset = EmojiInpaintingDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
    )

    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Setup scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare with accelerator
    if args.train_text_encoder:
        models = [unet, text_encoder]
    else:
        models = [unet]

    models, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        models, optimizer, train_dataloader, lr_scheduler
    )

    if args.train_text_encoder:
        unet, text_encoder = models
    else:
        unet = models[0]

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Log info
    print(f"\n=== Training Configuration ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Target steps: {args.max_train_steps}")
    print(f"Batch size: {args.train_batch_size}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Train conv layers: {args.train_conv_layers}")
    print(f"Train text encoder: {args.train_text_encoder}")
    print(f"Mixed precision: {args.mixed_precision}")
    print("=" * 30)

    # Training loop
    global_step = 0
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )

    # For tracking best model
    best_loss = float("inf")

    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Convert images to latents
                with torch.no_grad():
                    # Target latents (original image)
                    target_latents = vae.encode(
                        batch["orig_images"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    target_latents = target_latents * vae.config.scaling_factor

                    # Input latents (erased image)
                    erased_latents = vae.encode(
                        batch["erased_images"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    erased_latents = erased_latents * vae.config.scaling_factor

                # Resize mask to latent size
                mask = batch["masks"].to(dtype=weight_dtype)
                mask_resized = F.interpolate(
                    mask,
                    size=(args.resolution // 8, args.resolution // 8),
                    mode="nearest",
                )

                # Sample noise and timesteps
                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=target_latents.device,
                ).long()

                # Add noise to target latents
                noisy_latents = noise_scheduler.add_noise(
                    target_latents, noise, timesteps
                )

                # Concatenate for inpainting
                model_input = torch.cat(
                    [noisy_latents, mask_resized, erased_latents], dim=1
                )

                # Predict noise
                noise_pred = unet(model_input, timesteps, encoder_hidden_states).sample

                # Calculate loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Logging
                if global_step % 10 == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                        "epoch": epoch,
                    }
                    progress_bar.set_postfix(logs)

                    if accelerator.is_main_process:
                        accelerator.log(logs, step=global_step)

                # Save checkpoint
                if (
                    global_step % args.checkpointing_steps == 0
                    and accelerator.is_main_process
                ):
                    # Save best model
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_path = os.path.join(args.output_dir, "best_model")
                        os.makedirs(best_path, exist_ok=True)

                        # Save LoRA weights
                        unet.save_attn_procs(best_path)

                        # Save as safetensors
                        peft_state_dict = get_peft_model_state_dict(unet)
                        save_file(
                            peft_state_dict,
                            os.path.join(best_path, "pytorch_lora_weights.safetensors"),
                        )

                        print(
                            f"\nSaved best model (loss: {best_loss:.4f}) to {best_path}"
                        )

                    # Save regular checkpoint
                    checkpoint_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(checkpoint_path)
                    print(f"Saved checkpoint at step {global_step}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Save final model
    if accelerator.is_main_process:
        print(f"\nSaving final model to {args.output_dir}")

        # Save LoRA weights
        unet.save_attn_procs(args.output_dir)

        # Save as safetensors
        peft_state_dict = get_peft_model_state_dict(unet)
        save_file(
            peft_state_dict,
            os.path.join(args.output_dir, "pytorch_lora_weights.safetensors"),
        )

        # Save config
        import json

        config = {
            "pretrained_model": args.pretrained_model_name_or_path,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "train_steps": global_step,
            "final_loss": loss.item() if "loss" in locals() else None,
            "train_conv_layers": args.train_conv_layers,
            "train_text_encoder": args.train_text_encoder,
        }

        with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Training completed in {global_step} steps")
        print(f"Final model saved to {args.output_dir}")
        print(f"Best loss achieved: {best_loss:.4f}")


if __name__ == "__main__":
    main()
