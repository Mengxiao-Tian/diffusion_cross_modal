from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DiffusionPipeline, StableDiffusionImg2ImgPipeline
import cv2
import sys, os
import torch
import json
import numpy as np
from PIL import Image
import torchmetrics as tm
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from numpy import trace
from numpy import cov
from numpy import iscomplexobj
import numpy
from scipy.linalg import sqrtm

from torchmetrics.functional.multimodal import clip_score
from functools import partial
from torchvision.transforms import functional as F

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).unsqueeze(0).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)


def compute_torchmetric_fid(gen_imgs, gt_imgs, prompts):
    #print(gen_imgs, gt_imgs)
    #exit(0)

    real_images = [np.array(Image.open(path).convert("RGB").resize((768, 512))) for path in gt_imgs]
    fake_images = [np.array(Image.open(path).convert("RGB").resize((768, 512))) for path in gen_imgs]

    # print(len(real_images), len(fake_images))
    # exit(0)

    clip_score_real = calculate_clip_score(real_images[0], prompts)
    clip_score_fake = calculate_clip_score(fake_images[0], prompts)

    def preprocess_image(image):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2) / 255.0
        return image

    real_images = torch.cat([preprocess_image(image) for image in real_images])
    fake_images = torch.cat([preprocess_image(image) for image in fake_images])
    #fake_images = torch.cat([preprocess_image(fake_images[i]) for i in range(0, len(fake_images), 5)])

    # torch
    fid = FrechetInceptionDistance(normalize=True)


    fid.update(real_images[0].unsqueeze(0).repeat(2, 1, 1, 1), real=True)
    fid.update(fake_images[0].unsqueeze(0).repeat(2, 1, 1, 1), real=False)

    val =  fid.compute().item()//2
    fid.reset()
    return val, clip_score_real, clip_score_fake



root_dir = '/data/Part2/tmx/VSRN_data/f30k_precomp/'
split = 'test'
local_images_original = []
local_images_generate = []
captions = []

all_splits = [split]

with open(root_dir+'dataset_flickr30k.json', 'r') as f:
    raw_data = json.load(f)
    data = {split:[x for x in raw_data['images'] if x['split'] == split] for split in all_splits}
    for split_line in all_splits:
        for img_id, image_with_caption in enumerate(data[split_line]):
            img_name = image_with_caption['filename']
            img_path = os.path.join(root_dir, 'f30k_raw/flickr30k-images/'+img_name)
            local_images_original.append(img_path)
            for cap in image_with_caption['sentences']:
                captions.append(cap['raw'])

with open('Generated_images.txt', 'r') as f:
    for line in f:
        local_images_generate.append(str(line.strip()))


# fid = compute_torchmetric_fid(local_images_generate, local_images_original)
# print(fid)

results = []

with open('results.txt', 'a+') as f:
    for i in range(len(local_images_original)):
        local_images_generate_1_5 = local_images_generate[i*5:i*5+5]
        captions_1_5 = captions[i*5:i*5+5]
        print(local_images_generate_1_5, local_images_original[i], captions_1_5)
        for j in range(len(local_images_generate_1_5)):
            fid, clip_real, clip_fake = compute_torchmetric_fid([local_images_generate_1_5[j]], [local_images_original[i]], captions_1_5[j])
            print(fid, clip_real, clip_fake)
            f.write(str(fid) + ' ' + str(clip_real) + ' ' + str(clip_fake) + '\n')
            f.flush()
     