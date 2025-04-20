
from dataset import ShadowDataset
from model import ShadowGenerator, SegNetCNN
import config
import torch
import os
import random
from collections import OrderedDict
import torch.nn.functional as F # Needed for sigmoid if not applied in model
from helpers import display_predictions
import transform

# Keep load_generator if you might use it later, otherwise it can be removed
# def load_generator(model_path): ...

# --- New function to display SegNet output ---

# Keep display_images if you might use it for the GAN later
# def display_images (tuples): ...

def load_model(model_class, model_path: str, device, *args, **kwargs) -> torch.nn.Module:
    model = model_class(*args, **kwargs).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit()
    
    state_dict = torch.load(model_path, map_location=device)
    
    if next(iter(state_dict)).startswith('module.'):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Loaded state dict (removed 'module.' prefix).")
    else:   
        print("Loaded state dict.")
        
    model.load_state_dict(state_dict)
    model.eval()
    return model

def run_inference(model, dataset, device, indices):
    results = []
    with torch.no_grad():
        for i in indices:
            image, gt_mask = dataset[i]
            image_batch = image.unsqueeze(0).to(device)

            prediction_logits = model(image_batch)
            prediction_probs = torch.sigmoid(prediction_logits)

            prediction_cpu = prediction_probs.squeeze(0).cpu()
            thresholded_mask = (prediction_cpu > 0.5).float()

            results.append((image.cpu(), thresholded_mask, gt_mask.cpu()))
    return results


if __name__ == '__main__':

    test_set = ShadowDataset(
        config.TEST_IMAGE_DATASET_PATH,
        config.TEST_MASK_DATASET_PATH,
        config.TEST_REMOVED_DATASET_PATH,
        image_transform=transform.RGB_display,
        mask_transform=transform.mask_display ,
        mode='mask'
    )

    print(f"Loaded test dataset with {len(test_set)} images.")

    device = config.DEVICE
    model = load_model(SegNetCNN, os.path.join(config.MODEL_OUTPUT_DIR, 'segnet.pth'), device)

    num_images_to_display = 5
    print(f"Displaying SegNet output for {num_images_to_display} random test images...")

    tuple_list = []
    
    random_indices = random.sample(range(len(test_set)), num_images_to_display)

    tuple_list = run_inference(model, test_set, device, random_indices)

    display_predictions(tuple_list)

    print("Finished displaying images.")
