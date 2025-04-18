from dataset import ShadowDetectionDataset
from model import ShadowGenerator
import config
from torchvision import transforms
import torch
import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt

def load_generator(model_path):
    G = ShadowGenerator()
    G.load_state_dict(torch.load(model_path))
    return G

def display_images (tuples):
    """
    Display the input image, ground truth mask, and generated mask side by side.

    Args:
        tuples (list): A list of tuples containing the input image, ground truth mask, and generated mask.

    """
    for (image, mask, output) in (tuples):
        _, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(image.squeeze().transpose(1,2,0))
        ax[0].set_title('Input Image')
        ax[0].axis('off')
        ax[1].imshow(mask.squeeze(), cmap='gray')
        ax[1].set_title('Ground Truth Mask')
        ax[1].axis('off')
        ax[2].imshow(output.squeeze(), cmap='gray')
        ax[2].set_title('Generated Mask')
        ax[2].axis('off')
        plt.show()

if __name__ == '__main__':
    test_set = ShadowDetectionDataset(config.TEST_IMAGE_DATASET_PATH, config.TEST_MASK_DATASET_PATH, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((config.IMAGE_SIZE,config.IMAGE_SIZE)),transforms.Normalize(mean=(0.5,), std=(0.5,))]))
    print()
    G = load_generator(os.path.join(config.MODEL_OUTPUT_DIR, 'generator.pth'))
    device = config.DEVICE
    G = G.to(device)
    G.eval()

    tuples=[]


    for i in range(10):
        index = math.floor(random.random()*len(test_set))
        image, mask = test_set[index]
        image = image.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        output = G(image)
        tuples.append((image.cpu().detach().numpy(), mask.cpu().detach().numpy(), output.cpu().detach().numpy()))


    display_images(tuples)