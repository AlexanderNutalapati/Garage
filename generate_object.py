from models.flux.src.flux.cli import main as flux
from models.da_v2.run import main as depth_anything_v2

import numpy as np
import torch
from PIL import Image
import json
import os

import torch
from diffusers import FluxPipeline

batch_size = 5

with open('/home/jovyan/afilatov/research_for_gen_aug/Garage/objects.json', 'r') as file:
    objects = json.load(file)

#object = objects[list(objects.keys())[0]][0] #TODO: Implement prompts for objects, search for better prompts for flux
objects_keys = objects.keys()
root_path = "/home/jovyan/afilatov/research_for_gen_aug/generated_objects"

device = "cuda"
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

for object_key in objects_keys:
    object_path = root_path + '/' + object_key
    images_path = object_path + '/images'
    os.mkdir(object_path)
    os.mkdir(images_path)

    object = objects[object_key]
    outputs = []
    for batch in range(0,len(object),batch_size):
        if batch + batch_size < len(objects[object_key]):
            prompts = object[batch:batch+5]
        else:
            prompts = object[batch:-1]
        out = pipe(
                prompt=prompts,
                guidance_scale=3.5,
                height=768,
                width=1360,
                num_inference_steps=50,
            ).images
        outputs += out
    
    for i in range(len(outputs)):
        prompted_image_path = images_path + f"/prompt{i:05}"
        
        
        
            
    
    
    for i,prompt in enumerate(objects[object_key]):
        print(prompt)
        image_dir = generate_object(prompt, images_path, i)
        prompts[prompt] = image_dir
    with open(f'{object_path}/prompts.json', 'w') as f:
        json.dump(prompts, f)

