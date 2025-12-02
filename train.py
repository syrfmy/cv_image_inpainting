#!/usr/bin/env python3
"""
Stable Diffusion LoRA Training - Adapted for your dataset structure
Updated to use PEFT library instead of deprecated LoRAAttnProcessor
"""

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler

# PEFT imports
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

logger = get_logger(__name__)


def prepare_mask_and_masked_image(image, mask):
    """Prepare mask and masked image for inpainting"""
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA Training for Inpainting using PEFT"
    )

    # Required arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
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
        default="lora-inpainting-model",
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
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
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
        default=5e-6,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam."
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )

    # LoRA parameters
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA layers.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Alpha parameter for LoRA scaling.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="Dropout probability for LoRA layers.",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        choices=["none", "all", "lora_only"],
        help="Bias type for LoRA. 'none': no bias, 'all': all bias, 'lora_only': only LoRA bias",
    )

    # System parameters
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    args = parser.parse_args()

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


class InpaintingDataset(Dataset):
    """
    Dataset for inpainting training with your folder structure.
    Expects: train_data_dir/erased/, train_data_dir/masks/, train_data_dir/orig/
    """

    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.data_root = Path(data_root)

        # Load paths from your dataset structure
        self.erased_dir = self.data_root / "erased"
        self.masks_dir = self.data_root / "masks"
        self.orig_dir = self.data_root / "orig"

        # Verify folders exist
        assert self.erased_dir.exists(), f"erased folder not found: {self.erased_dir}"
        assert self.masks_dir.exists(), f"masks folder not found: {self.masks_dir}"
        assert self.orig_dir.exists(), f"orig folder not found: {self.orig_dir}"

        # Get all erased files
        self.erased_files = sorted(list(self.erased_dir.glob("*_erased.png")))

        print(f"Found {len(self.erased_files)} training samples")
        if len(self.erased_files) > 0:
            print(f"Example: {self.erased_files[0].name}")

        # Image transforms
        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.erased_files)

    def __getitem__(self, index):
        # Get paths
        erased_path = self.erased_files[index]

        # Construct corresponding mask and original paths
        filename = erased_path.stem  # e.g., "6__val_m00_erased"
        base_name = filename.replace("_erased", "")  # e.g., "6__val_m00"

        mask_path = self.masks_dir / f"{base_name}_mask.png"
        orig_path = self.orig_dir / f"{base_name}.png"

        # Extract emoji_id for prompt
        emoji_id = base_name.split("__")[0]
        prompt = f"emoji {emoji_id}"

        # Load images
        erased_image = Image.open(erased_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")
        orig_image = Image.open(orig_path).convert("RGB")

        # Apply transforms
        erased_image = self.image_transforms_resize_and_crop(erased_image)
        orig_image = self.image_transforms_resize_and_crop(orig_image)

        # Tokenize prompt
        prompt_ids = self.tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return {
            "erased_image": erased_image,  # Will be used as input
            "orig_image": orig_image,  # Will be used as target
            "mask": mask_image,  # Will be used for inpainting
            "prompt_ids": prompt_ids,
            "prompt": prompt,
        }


class CollateFn:
    """Collate function class that holds the tokenizer"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        """Collate function for the dataloader"""
        input_ids = [example["prompt_ids"] for example in examples]
        pixel_values = [
            example["orig_image"] for example in examples
        ]  # Use original as target

        # Convert images to tensors
        pixel_values = [transforms.ToTensor()(img) for img in pixel_values]
        pixel_values = [transforms.Normalize([0.5], [0.5])(img) for img in pixel_values]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        # Prepare masks and masked images (erased images)
        masks = []
        masked_images = []

        for example in examples:
            orig_pil = example["orig_image"]
            erased_pil = example["erased_image"]
            mask_pil = example["mask"]

            # Resize mask to match image size
            mask_pil = mask_pil.resize(orig_pil.size, Image.NEAREST)

            # Prepare mask and masked image
            mask, masked_image = prepare_mask_and_masked_image(orig_pil, mask_pil)

            masks.append(mask)
            masked_images.append(masked_image)

        masks = torch.stack(masks)
        masked_images = torch.stack(masked_images)

        # Pad input_ids
        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        ).input_ids

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "masks": masks,
            "masked_images": masked_images,
        }


def main():
    args = parse_args()

    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )

    # Load models
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    # Freeze models - only train LoRA
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Set dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to device and set dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Enable xformers if requested
    if args.enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available

        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available")

    # Setup LoRA using PEFT
    # Create LoRA config
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )

    # Add LoRA adapter to UNet
    unet.add_adapter(lora_config)

    # Now only LoRA parameters are trainable
    unet.train()

    # Get trainable parameters
    trainable_params = []
    for name, param in unet.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    print(f"Number of trainable parameters: {len(trainable_params)}")

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Setup optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes: pip install bitsandbytes")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Load dataset
    train_dataset = InpaintingDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=False,
    )

    # Create collate function
    collate_fn_instance = CollateFn(tokenizer)

    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn_instance,
        num_workers=2,
    )

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Setup scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Training setup
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  LoRA rank = {args.lora_rank}")
    logger.info(f"  LoRA alpha = {args.lora_alpha}")

    # Training loop
    global_step = 0
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(args.num_train_epochs):
        unet.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Convert masked images to latent space
                masked_latents = vae.encode(
                    batch["masked_images"]
                    .reshape(batch["pixel_values"].shape)
                    .to(dtype=weight_dtype)
                ).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor

                # Resize masks to latent shape
                masks = batch["masks"]
                mask = torch.stack(
                    [
                        torch.nn.functional.interpolate(
                            mask, size=(args.resolution // 8, args.resolution // 8)
                        )
                        for mask in masks
                    ]
                ).to(dtype=weight_dtype)
                mask = mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8)

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample timesteps
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Concatenate for inpainting
                latent_model_input = torch.cat(
                    [noisy_latents, mask, masked_latents], dim=1
                )

                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict noise
                noise_pred = unet(
                    latent_model_input, timesteps, encoder_hidden_states
                ).sample

                # Calculate loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = trainable_params
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Save checkpoint
                if (
                    global_step % args.checkpointing_steps == 0
                    and accelerator.is_main_process
                ):
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

            # Log
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Save final model
    if accelerator.is_main_process:
        # Save LoRA weights
        unet = unet.to(torch.float32)

        # Save using PEFT's method
        from peft import set_peft_model_state_dict

        # Get the state dict
        peft_state_dict = get_peft_model_state_dict(unet)

        # Save LoRA weights
        lora_save_path = os.path.join(args.output_dir, "lora_weights.safetensors")
        torch.save(peft_state_dict, lora_save_path)

        # Also save in the diffusers format for compatibility
        unet.save_attn_procs(args.output_dir)

        logger.info(f"Saved LoRA weights to {args.output_dir}")
        logger.info(f"Saved PEFT LoRA weights to {lora_save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
