from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DiffusionPipeline, StableDiffusionImg2ImgPipeline
import cv2
import sys, os
import torch
import json
import numpy as np
from PIL import Image
import torchmetrics as tm
from tqdm import tqdm

torch.cuda.set_device(1)


'''
model_path = './promptnet1/checkpoint-200000'
unet = UNet2DConditionModel.from_pretrained(model_path+"/unet")
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", unet=unet, torch_dtype=torch.float32)
pipe.to("cuda")
'''

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float32, local_files_only=True)
pipe = pipe.to("cuda")
vae = pipe.vae
unet = pipe.unet
text_encoder = pipe.text_encoder

# Freeze vae and text_encoder
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)


'''
model_path = './promptnet1/checkpoint-200000'
unet = UNet2DConditionModel.from_pretrained(model_path+"/unet")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", unet=unet, torch_dtype=torch.float32)
pipe = pipe.to("cuda")
'''


root_dir = '/data/Part2/tmx/VSRN_data/f30k_precomp/'
split = 'test'
local_images = []
captions = []
all_splits = [split]

with open(root_dir+'dataset_flickr30k.json', 'r') as f:
    raw_data = json.load(f)
    data = {split:[x for x in raw_data['images'] if x['split'] == split] for split in all_splits}
    for split_line in all_splits:
        for img_id, image_with_caption in enumerate(data[split_line]):

            img_name = image_with_caption['filename']
            img_path = os.path.join(root_dir, 'f30k_raw/flickr30k-images/'+img_name)
            local_images.append(img_path)

            for cap in image_with_caption['sentences']:
                captions.append(cap['raw'])

local_images = local_images[:100]
captions = captions[:500]

kk = 0
local_images_my = []
for i in tqdm(range(len(local_images))):
    img_paths = local_images[i]  
    imgs = Image.open(f'{img_paths}').convert("RGB")
    imgs = imgs.resize((768, 512))
    prompts = captions[i*5:i*5+5]
    # save_path = os.path.join('img', str(i))
    # if os.path.exists(save_path)==False:
    #     os.mkdir(save_path)
    imgs.save(os.path.join('Generated_images', f"validation_original_{str(i)}.png"))
    for prompt in prompts:  
        print(prompt)
        image = pipe(prompt, num_inference_steps=30, guidance_scale=8.5).images[0]
        #image = pipe(prompt=prompt, init_image=imgs, strength=0.75, guidance_scale=7.5).images[0]

        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #image = cv2.putText(
        #    image, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
        #)
        kk_str = str(kk)
        save_path_new = os.path.join('Generated_images', f"validation_new_{kk_str}.png")
        local_images_my.append(save_path_new)
        cv2.imwrite(save_path_new, image)
        kk = kk + 1

with open('Generated_images.txt', 'w') as f:
    for line in local_images_my:
        f.write(line+'\n')

     