import requests
import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image, DONT FORGET TO WRITE YOUR IMAGE NAME
#img_path = "game-bg.jpg"
# convert it into an RGB format 

def caption_image(input_image: np.ndarray):
    image = Image.fromarray(input_image).convert('RGB')

    text = "the image of"
    inputs = processor(images=image, text=text, return_tensors="pt")

    outputs = model.generate(**inputs, max_length=50)

    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)


iface.launch(server_name="0.0.0.0", server_port= 7860)
