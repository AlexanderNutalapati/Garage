from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

prompts = ["a cup"]
image = Image.open('/home/jovyan/afilatov/research_for_gen_aug/Garage/object_raw_image.jpg')

print(np.array(image).shape)


inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")

with torch.no_grad():
  outputs = model(**inputs)

preds = outputs.logits.unsqueeze(1)
print(preds.shape)

filename = f"mask.png"
# here we save the second mask
plt.imsave(filename,torch.sigmoid(preds[0][0]))
"""
import cv2

img2 = cv2.imread(filename)
resized_img = cv2.resize(img2, (1360, 760))
cv2.imwrite('/home/jovyan/afilatov/research_for_gen_aug/Garage/mask.png', resized_img)

gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

(thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

# fix color format
cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)

Image.fromarray(bw_image)
"""