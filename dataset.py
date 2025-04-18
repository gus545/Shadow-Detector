from torch.utils.data import Dataset
import cv2
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

class ShadowDetectionDataset(Dataset):
    """
    A custom dataset class for shadow detection.

    Args:
        image_dir (str): The directory path containing the input images.
        mask_dir (str): The directory path containing the corresponding masks.

    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = os.listdir(image_dir)
        self.mask_files = os.listdir(mask_dir)
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of images in the dataset.

        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns the image and mask at the given index.

        Args:
            idx (int): The index of the image and mask to retrieve.

        Returns:
            tuple: A tuple containing the image and mask.

        """
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def get_train_and_val(image_path, mask_path, batch_size, image_size, num_workers=4):

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((image_size, image_size)),
                                    transforms.Normalize(mean=[0.5], std=[0.5]),
                                    ])
    
    
    dataset = ShadowDetectionDataset(image_path, mask_path, transform=transform)

    #split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, val_dataset


