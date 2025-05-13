import torch
import numpy as np
from PIL import Image
import gradio as gr
from pathlib import Path
from torchvision import transforms

from model import VisionGPT2Model
from transformers import GPT2TokenizerFast
from config import config

# Load model
def load_model():
    model = VisionGPT2Model(config)
    model.load_state_dict(torch.load("captioner.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Inference function
def generate_caption(image):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    sequence = torch.ones(1, 1).long() * tokenizer.bos_token_id
    
    with torch.no_grad():
        caption_ids = model.generate(
            image_tensor,
            sequence,
            max_tokens=50,
            temperature=1.0,
            deterministic=True
        )
    
    caption = tokenizer.decode(caption_ids.numpy(), skip_special_tokens=True)
    return caption

# Gradio App
interface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Caption Generator (ViT + GPT-2)",
    description="Upload an image and generate a caption using a hybrid Vision Transformer + GPT-2 model."
)

interface.launch()
