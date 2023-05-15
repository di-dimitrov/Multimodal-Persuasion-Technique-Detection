import torch 
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


def process_text(text):
    text_inputs = tokenizer(text, padding=True, return_tensors="pt")
    text_features = model.get_text_features(**text_inputs)


def process_image(img_path):
    image = Image.open(img_path)

    inputs = processor(images=image, return_tensors="pt")

    image_features = model.get_image_features(**inputs)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(768, 22)
       

    def forward(self, x):
        x = self.flatten(x)
        logits = self.projection(x)
        return logits