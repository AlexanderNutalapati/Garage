import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Function to load an image from a URL
def load_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Load the model and processor
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Load an example image (you can replace this with your own image path or URL)
image_url = "https://example.com/path/to/your/image.jpg"
original_image = load_image(image_url)

# Define the target size for processing (adjust as needed)
target_size = (512, 512)

# Resize the image for processing
resized_image = original_image.resize(target_size, Image.LANCZOS)

# Prepare the text prompt
prompts = ["a dog"]

# Process the image and text
inputs = processor(text=prompts, images=[resized_image], padding="max_length", return_tensors="pt")

# Generate the mask
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted mask
mask = outputs.logits.sigmoid()[0]

# Convert the mask to a numpy array and resize to match the original image
mask_np = mask.numpy()
mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
high_res_mask = mask_image.resize(original_image.size, Image.LANCZOS)

# Save or display the high-resolution mask
high_res_mask.save("high_res_mask.png")
high_res_mask.show()

# Optionally, apply the mask to the original image
masked_image = Image.composite(original_image, Image.new('RGB', original_image.size, 'black'), high_res_mask)
masked_image.save("masked_image.png")
masked_image.show()