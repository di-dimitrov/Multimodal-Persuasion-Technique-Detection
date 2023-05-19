import torch 
import torchvision
import torch.nn as nn

class CLIPFeatureFusion(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(768, 22)

    def forward(self, text_feat, image_feat):
        x = torch.cat(text_feat,image_feat)
        x = self.flatten(x)
        logits = self.projection(x)
        return logits