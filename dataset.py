from torch.utils.data import Dataset
import os

import cv2

class ShadowDataset(Dataset):

    def __init__(self, image_dir, mask_dir, removed_dir, image_transform=None, mask_transform=None, mode='full'):
        self.image_paths = self.get_paths(image_dir)
        self.mask_paths = self.get_paths(mask_dir)
        self.removed_paths = self.get_paths(removed_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.mode = mode
    def get_paths(self, dir):
        return [os.path.join(dir, f) for f in sorted(os.listdir(dir))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        shadow_img = cv2.imread(self.image_paths[idx])
        mask_img = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.imread(self.removed_paths[idx])

        shadow_img = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2RGB)
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)


        if self.image_transform:
            shadow_img = self.image_transform(shadow_img)
            clean_img = self.image_transform(clean_img)
        if self.mask_transform:
            mask_img = self.mask_transform(mask_img)

        if self.mode == 'mask':
            return shadow_img, mask_img
        elif self.mode == 'removal':
            return shadow_img, mask_img, clean_img
        elif self.mode == 'full':
            return {
                'shadow_img': shadow_img,
                'mask': mask_img,
                'clean_img': clean_img
            }
        else:
            raise ValueError(f"Invalid mode {self.mode}")
