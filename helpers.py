import matplotlib.pyplot as plt
from typing import List, Tuple
import torch
import numpy as np

def display_predictions(img_tuples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], title: str = None):
    """
    Display comparison of input images, prediction images and ground truth images in each row

    Args:
        img_tuples (List[Tuple[Tensor, Tensor, Tensor]]): list of tuples (input, prediction, ground_truth)
        title (str): optional title for the plot
    """
    num_samples = len(img_tuples)
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if title:
        fig.suptitle(title, fontsize=16)

    for i, (input_img, pred_image, true_image) in enumerate(img_tuples):
        (input_img, pred_mask, true_mask) = prep_tensors_for_display((input_img, pred_image, true_image))

        # Handle axs shape for single row
        if num_samples == 1:
            ax_input, ax_pred, ax_true = axs[0], axs[1], axs[2]
        else:
            ax_input, ax_pred, ax_true = axs[i]

        # Convert input image to (H, W, C) for color
        if isinstance(input_img, torch.Tensor):
            input_img = input_img.permute(1, 2, 0).cpu().numpy()

        ax_input.imshow(input_img)
        ax_pred.imshow(pred_mask, cmap='gray')
        ax_true.imshow(true_mask, cmap='gray')

        ax_input.axis('off')
        ax_pred.axis('off')
        ax_true.axis('off')

        # Add column titles only on the first row
        if i == 0:
            ax_input.set_title("Input Image")
            ax_pred.set_title("Predicted Mask")
            ax_true.set_title("Ground Truth")

    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.92)
    plt.show()

def prep_tensors_for_display(img_tuple : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Prepares tensors for display by moving them to CPU, converting to numpy arrays,
    and adjusting dimensions and ranges for display.

    Args:
        img_tuple (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple of tensors (input_img, pred_mask, true_mask).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of numpy arrays ready for display.
    """
    out_tuple = []
    for tensor in img_tuple:        
        tensor = tensor.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
            tensor = (tensor * 0.5) + 0.5
            tensor = np.clip(tensor, 0, 1)
        tensor = tensor.numpy()
        if tensor.ndim == 3 and tensor.shape[2] == 1:
            tensor = tensor.squeeze(2)

        out_tuple.append(tensor)
    
    return out_tuple
