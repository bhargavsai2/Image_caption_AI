import requests
import gradio as gr
import numpy as np
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# URL of the page to scrape
url = "https://en.wikipedia.org/wiki/IBM"

response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

img_elements = soup.find_all('img')
#print(img_elements)

with open("captions.txt", "w") as caption_file:

    for img_element in img_elements:
        img_url = img_element.get('src')

        if 'svg' in img_url or '1X1' in img_url:
            continue
        
        if img_url.startswith('//'):
            img_url = 'http:' + img_url
        elif not img_url.startswith('http://') and not img_url.startswith('https://'):
            continue

        
        try:
            # Download the image
            response = requests.get(img_url)
            # Convert the image data to a PIL Image
            raw_img = Image.open(BytesIO(response.content))
            if raw_img.size[0] * raw_img.size[1] < 400:
                continue
            
            raw_img = raw_img.convert('RGB')
            
            # Process the image
            inputs = processor(raw_img, return_tensors="pt")
            # Generate a caption for the image
            out = model.generate(**inputs, max_new_tokens=50)
            # Decode the generated tokens to text
            caption = processor.decode(out[0], skip_special_tokens=True)
            # Write the caption to the file, prepended by the image URL
            caption_file.write(f"{img_url}: {caption}\n")
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            continue

