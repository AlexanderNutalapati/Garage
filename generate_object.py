# from models.flux.src.flux.cli import main as flux
# from models.da_v2.run import main as depth_anything_v2

import numpy as np
import torch
from PIL import Image
import json
import os
from transformers import pipeline
import torch
from diffusers import FluxPipeline
import gc
from torchvision import transforms
from transformers import AutoModelForImageSegmentation





batch_size = 5

with open('/home/jovyan/afilatov/research_for_gen_aug/Garage/objects_prompts.json', 'r') as file:
    objects = json.load(file)

#object = objects[list(objects.keys())[0]][0] #TODO: Implement prompts for objects, search for better prompts for flux
objects_keys = objects.keys()
root_path = "/home/jovyan/afilatov/research_for_gen_aug/generated_objects"


def extract_object(birefnet, imagepath):
        # Data settings
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(imagepath)
        input_images = transform_image(image).unsqueeze(0).to('cuda')

        # Prediction
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
        return image, mask



for object_key in objects_keys:
    object_path = root_path + '/' + object_key
    images_path = object_path + '/images'
    os.mkdir(object_path)                                 # NOTE:UNCOMMENT THIS AND NEXT LINE
    os.mkdir(images_path)

    device = "cuda"


    flux = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    flux = flux.to(device)
    object = objects[object_key]
    outputs = []
    for batch in range(0,len(object),batch_size):
        if batch + batch_size < len(objects[object_key]):
            prompts = object[batch:batch+5]
        else:
            prompts = object[batch:-1]
        out = flux(
                prompt=prompts,
                guidance_scale=3.5,
                height=768,
                width=1360,
                num_inference_steps=50,
            ).images
        outputs += out
        # if batch+batch_size > 2:
        #     break                                                                        #NOTE: REMOVE THIS BREAK!
    
    prompts = {}
    for i in range(len(outputs)):
        prompted_image_path = images_path + f"/prompt{i:05}"
        os.mkdir(prompted_image_path)
        outputs[i].save(f"{prompted_image_path}/object_raw_image.jpg")
        prompts[f"prompt{i:05}"] = object[i]
    with open(f'{object_path}/prompts.json', 'w') as f:
        json.dump(prompts, f)


    print('nvidia-smi!!!!!')

    del flux
    gc.collect()
    torch.cuda.empty_cache()

    
    
    ls = os.listdir(images_path)
    images = []
    for prompt in ls:
        image = Image.open(images_path + '/'+ prompt + '/' + 'object_raw_image.jpg')
        images.append(image)


    depth_estimator = pipeline("depth-estimation", model="LiheYoung/depth-anything-large-hf", device = device)

    outputs = []
    for batch in range(0,len(images),batch_size):
        if batch + batch_size < len(objects[object_key]):
            images_batch = images[batch:batch+5]
        else:
            images_batch = images[batch:-1]
        out = depth_estimator(images_batch)
        outputs += out
        # if batch+batch_size > 2:
        #     break                                                                  #NOTE: REMOVE THIS BREAK!
    print(outputs[0].keys())

    for i in range(len(outputs)):
        prompted_image_path = images_path + f"/prompt{i:05}"
        outputs[i]['depth'].save(f"{prompted_image_path}/depth.jpg")
        
    del depth_estimator
    gc.collect()
    torch.cuda.empty_cache()
    

    birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    birefnet.to('cuda')
    birefnet.eval()
    batch_prompts = [object_key] * batch_size
    outputs = []
    print(ls)
    for i, prompt in enumerate(ls):
        prompted_image_path = images_path + f"/prompt{i:05}"
        _, mask = extract_object(birefnet, imagepath=f'{prompted_image_path}/object_raw_image.jpg')
        mask.save(f"{prompted_image_path}/mask.jpg")
     

    del birefnet
    gc.collect()
    torch.cuda.empty_cache()




    
    #break                                                                   #NOTE: REMOVE THIS BREAK!

        


