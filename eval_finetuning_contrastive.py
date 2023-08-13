# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import json
import argparse
import logging
import math
import os
import time
import random
from pathlib import Path
from typing import Optional
import cv2
import clip
import sys, os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from collections import defaultdict

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, StableDiffusionImg2ImgPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from accelerate import DistributedDataParallelKwargs

from datasets_loading_cross_domain import get_dataset
from utils import evaluate_scores, i2t, t2i

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='stabilityai/stable-diffusion-2-1-base',
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-unet_scoring",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--ignore_pretrained_unet",
        action="store_true"
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument("--task", type=str, default="flickr30k", help="The task to train on.")
    parser.add_argument('--neg_factor', type=float, default=0.0, help='The probability of sampling a negative image.')

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def calItr(img_embs, cap_embs, unet, vae, text_encoder, noise_scheduler, weight_dtype):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """

    print(img_embs.shape, cap_embs.shape)

    n_img = len(img_embs)
    n_cap = len(cap_embs)

    shard_size = 16 * 5


    t0 = time.time()
    n_im_shard = (n_img-1)//shard_size + 1
    n_cap_shard = (n_cap-1)//shard_size + 1
    d = np.zeros((n_img, n_cap))
    d_ids = np.zeros((n_img, n_cap))

    if sys.stdout.isatty():
        pbar = tqdm(total=(n_im_shard * n_cap_shard))

    t = torch.arange(start=0, end=1000).cuda()
    batchsize = 32
    t = t.split(batchsize, dim=0)

    beta = torch.linspace(0.1 / 1000, 20 / 1000, 1000)
    alpha = (1 - beta)
    alpha_bar = alpha.cumprod(dim=0).cuda()

    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), n_img)
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), n_cap)
            with torch.no_grad():
                img_block = torch.from_numpy(img_embs[im_start:im_end]).cuda()
                cap_block = torch.from_numpy(cap_embs[cap_start:cap_end]).cuda()

                latents = img_block
                encoder_hidden_states = cap_block

                y = []

                for text_idx, text in enumerate(encoder_hidden_states):

                    loss_a = []

                    for img_idx, img in tqdm(enumerate(latents)):
                        #print(text.shape, img.shape)
                        now_x = img.unsqueeze(0).contiguous()
                        now_y = text.unsqueeze(0).contiguous()

                        total_loss = 0
                        flag_jishi = 0

                        for index in range(len(t)):
                            if(flag_jishi >= 2):
                                break

                            flag_jishi = flag_jishi + 1

                            tensor_t = t[index]
                            size = tensor_t.shape[0]

                            now_x1 = now_x.repeat(size, 1, 1, 1)
                            now_y1 = now_y.repeat(size, 1, 1)

                            noise = torch.randn_like(now_x1)

                            noised_x = torch.sqrt(alpha_bar[tensor_t]).view(-1, 1, 1, 1) * now_x1 + \
                                    torch.sqrt(1 - alpha_bar[tensor_t]).view(-1, 1, 1, 1) * noise

                            target = noise

                            # Predict the noise residual and compute loss
                            model_pred = unet(noised_x, tensor_t, now_y1).sample
                            #model_pred = model_pred[:, :3, :, :]
                            #target = target[:, :3, :, :]
                            loss = torch.nn.MSELoss()(model_pred.float(), target.float())
                            loss = loss * tensor_t.shape[0] / batchsize
                            total_loss += loss 
                        
                        total_loss = total_loss / flag_jishi
                        loss_a.append(total_loss)

                    loss_a = torch.tensor(loss_a).to(latents.device)
                    loss_a = loss_a * -1
                    y.append(loss_a)

                y = torch.stack(y, dim=1).squeeze(0)
                #print(y.shape)
            
            #print(d[im_start:im_end, cap_start:cap_end].shape)

            print('loss', y)
            d[im_start:im_end, cap_start:cap_end] = y.data.cpu().numpy()
            if sys.stdout.isatty():
                pbar.update(1)
    if sys.stdout.isatty():
        pbar.close()
    print('Calculate similarity matrix elapses: {:.3f}s'.format(time.time() - t0))
    np.save('scores_f30k.npy', d)
    print('save successed!')
    return d

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo_name = create_repo(repo_name, exist_ok=True)
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    '''
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    model_path = './promptnet/'
    unet = UNet2DConditionModel.from_pretrained(model_path+"/unet")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", unet=unet, torch_dtype=torch.float32)
    pipe.to("cuda")

    image = pipe(prompt="yoda").images[0]

    # load the trained model
    save_path = './saved_model'
    scheduler = DDIMScheduler.from_pretrained(save_path, subfolder="scheduler")  # must use DDIM when refine_step > 0
    if use_fp16:
        pipe = StableDiffusionPromptNetPipeline.from_pretrained(save_path, scheduler=scheduler, torch_dtype=torch.float16)
        weight_dtype = torch.float16
    else:
        pipe = StableDiffusionPromptNetPipeline.from_pretrained(save_path, scheduler=scheduler)
        weight_dtype = torch.float32
    pipe.to("cuda")
    '''        

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    #unet.load_state_dict(torch.load('./promptnet_4090/unet_horizontalflip_continue.pt'))

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    def collate_fn(data):
        images, text, idx = zip(*data)
        pixel_values = torch.stack(images)
        ref_pixel_values = torch.stack(text, 0)
        ids = np.array(idx)
        return {"pixel_values": pixel_values, "ref_pixel_values": ref_pixel_values, "ids": ids}

    val_dataset = get_dataset(f'{args.task}', f'datasets/{args.task}', transform=None, split='test', tokenizer=tokenizer)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=8,
        num_workers=args.dataloader_num_workers,
    )

    val_dataloader = accelerator.prepare(val_dataloader)

    if accelerator.is_main_process:
        img = np.zeros((len(val_dataloader.dataset), 4, 16, 16), dtype=np.float32)
        cap = np.zeros((len(val_dataloader.dataset), 77, 1024), dtype=np.float32)
        ids_new = np.zeros((len(val_dataloader.dataset)))

        print(img.shape, cap.shape, ids_new.shape)
        
        metrics = []
        for k, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            imgs, text, ids = batch["pixel_values"], batch["ref_pixel_values"], batch["ids"]

            imgs = vae.encode(imgs.to(dtype=weight_dtype)).latent_dist.sample()
            imgs = imgs * 0.18215
            text = text_encoder(text.long())[0]

            img[ids] = imgs.detach().cpu().numpy().copy()
            cap[ids] = text.detach().cpu().numpy().copy()
            ids_new[ids] = ids
            del batch 
        
        img = np.array([img[i] for i in range(0, len(img), 5)])
        scores = calItr(img, cap, unet, vae, text_encoder, noise_scheduler, weight_dtype)
        np.save('scores_f30k1.npy', scores)
        np.save('ids_new1.npy', ids_new)

        r, rt = i2t(scores, return_ranks=True)
        ri, rti = t2i(scores, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

        '''
        scores = np.load('scores_f30k1.npy')
        ids_new = np.load('ids_new1.npy')

        r1s = []
        r5s = []
        max_more_than_onces = 0

        ids_new = ids_new[:500]

        scores_new = np.zeros((scores.shape[1], scores.shape[1]), dtype=np.float32)

        print(scores_new.shape)

        for i in range(len(scores_new)):
            left = i * 5
            right = i * 5 + 5
            if(right > 500):
                break
            for j in range(left, right):
                scores_new[j][:500] = scores[i]

        r1s,r5s= evaluate_scores(args, scores, ids_new)
        r1 = 100 * sum(r1s) / len(r1s) 
        r5 = 100 * sum(r5s) / len(r5s)
        print(f'R@1: {r1}')
        print(f'R@5: {r5}')
        '''

if __name__ == "__main__":
    main()