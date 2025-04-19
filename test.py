from dataset import ShadowDataset
from model import ShadowGenerator
import config
from torchvision import transforms
import torch
import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from collections import OrderedDict
from model import ShadowGenerator # Make sure ShadowGenerator is imported

def load_generator(model_path):
    """
    Loads the generator model state dictionary from the specified path,
    handling the potential 'module.' prefix from DataParallel.
    """
    print(f"Loading generator state dict from: {model_path}")
    
    # Load the checkpoint from the specified path.
    # Using map_location='cpu' is generally safer as it allows loading
    # a GPU-trained model onto a CPU machine if needed. The model will
    # be moved to the correct device later.
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Check if keys start with 'module.' (indicating it was saved from DataParallel)
    # and create a new state dict without the prefix if necessary.
    new_state_dict = OrderedDict()
    needs_stripping = any(key.startswith("module.") for key in state_dict.keys())

    for k, v in state_dict.items():
        if needs_stripping:
            # Remove 'module.' prefix
            new_key = k.replace("module.", "", 1) 
        else:
            new_key = k
        new_state_dict[new_key] = v

    # Create a new generator instance
    G = ShadowGenerator()
    
    # Load the (potentially modified) state dictionary into the model instance
    G.load_state_dict(new_state_dict)
    
    print("Generator loaded successfully.")
    return G


def display_images (tuples):
    """
    Display the input image, ground truth mask, and generated mask side by side.

    Args:
        tuples (list): A list of tuples containing the input image, ground truth mask, and generated mask.

    """
    for (image, mask, output) in (tuples):
        _, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(((image.squeeze().transpose(1,2,0) + 1) / 2).clip(0,1))
        ax[0].set_title('Input Image')
        ax[0].axis('off')
        ax[1].imshow(((mask.squeeze() + 1) / 2).clip(0,1), cmap='gray')
        ax[1].set_title('Ground Truth Mask')
        ax[1].axis('off')
        ax[2].imshow(((output.squeeze() + 1) / 2).clip(0,1), cmap='gray')
        ax[2].set_title('Generated Mask')
        ax[2].axis('off')
        plt.show()





if __name__ == '__main__':
    test_set = ShadowDataset(config.TEST_IMAGE_DATASET_PATH, config.TEST_MASK_DATASET_PATH, config.TEST_REMOVED_DATASET_PATH, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((config.IMAGE_SIZE,config.IMAGE_SIZE)),transforms.Normalize(mean=(0.5,), std=(0.5,))]), mode='mask')
    
    # Construct the full path to the model
    generator_model_path = os.path.join(config.MODEL_OUTPUT_DIR, 'generator.pth') # Use os.path.join for robustness
    
    # Load the generator using the corrected function
    G = load_generator(generator_model_path) 

    device = config.DEVICE
    G = G.to(device) # Move the loaded model to the desired device
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