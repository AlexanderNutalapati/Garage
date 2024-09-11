from models.flux.src.flux.cli import main as flux
from models.da_v2.run import main as depth_anything_v2

import numpy as np
import torch
from PIL import Image
import json

def generate_object():
    with open('/home/jovyan/afilatov/research_for_gen_aug/Garage/objects.json', 'r') as file:
        objects = json.load(file)
    object = list(objects.keys())[0] #TODO: Implement prompts for objects, search for better prompts for flux

    generated_object_image = flux(name = "flux-dev",
        prompt=object,
        device= "cuda" if torch.cuda.is_available() else "cpu",
        output_dir = "/home/jovyan/afilatov/research_for_gen_aug/Garage",)

    generated_object_image.save("/home/jovyan/afilatov/research_for_gen_aug/Garage/object_raw_image.jpg")
    generated_object_image = np.array(generated_object_image)

    depth = depth_anything_v2(generated_object_image,
            input_size=518,
            encoder='vitl')

    depth = Image.fromarray(depth)
    depth.save('/home/jovyan/afilatov/research_for_gen_aug/Garage/object_depth.jpg')

generate_object()

