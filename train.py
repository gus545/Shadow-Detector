import config
from model import ShadowGenerator,ShadowDiscriminator,init_weights
from dataset import ShadowDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
from torchvision.utils import save_image
from torchvision.utils import make_grid



def plot_losses(losses, title):
    for model in losses:
        plt.plot(losses[model], label=model)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(config.LOG_OUTPUT_DIR + '/losses_plot.png')
    plt.close()

def make_dir():
    if not os.path.exists(config.BASE_OUTPUT_DIR):
        os.makedirs(config.BASE_OUTPUT_DIR)
    if not os.path.exists(config.MODEL_OUTPUT_DIR):
        os.makedirs(config.MODEL_OUTPUT_DIR)
    if not os.path.exists(config.IMAGE_OUTPUT_DIR):
        os.makedirs(config.IMAGE_OUTPUT_DIR)
    if not os.path.exists(config.CHECKPOINT_OUTPUT_DIR):
        os.makedirs(config.CHECKPOINT_OUTPUT_DIR)
    if not os.path.exists(config.LOG_OUTPUT_DIR):
        os.makedirs(config.LOG_OUTPUT_DIR)

def get_train_and_val(image_dir, mask_dir, removed_dir, batch_size, image_size, num_workers, mode='full'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    dataset = ShadowDataset(image_dir, mask_dir, removed_dir, transform=transform, mode='mask')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader


def train_gan(generator, discriminator, dataloader, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn, num_epochs, device):
    
    generator.train()
    discriminator.train()

    g_losses = []
    d_losses = []

    try:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            pbar = tqdm(dataloader, desc="Training", leave=False)
            for real_images, masks in pbar:
                batch_size = real_images.size(0)
                real_images = real_images.to(device)

                # Train the discriminator

                #Reset gradients    
                d_optimizer.zero_grad()
                
                real_labels = torch.ones(batch_size, 1).to(device)
                real_outputs = discriminator(masks)
                d_loss_real = d_loss_fn(real_outputs, masks)
                
                # Generate fake images
                fake_images = generator(real_images)
                fake_labels = torch.zeros(batch_size, 1).to(device)

                #Evaluate discriminator
                fake_outputs = discriminator(fake_images.detach())
                d_loss_fake = d_loss_fn(fake_outputs, fake_labels)

                
                d_loss = d_loss_real + d_loss_fake

                # Update weights
                d_loss.backward()
                d_optimizer.step()

                #Progress tracking
                d_loss_val = d_loss.item()  # Assuming you calculated loss
                pbar.set_postfix({"Discriminator loss": d_loss_val})
                d_losses.append(d_loss.item())

                # Train Generator

                #Reset gradients
                g_optimizer.zero_grad()

                #Evaluate fake images
                fake_outputs = discriminator(fake_images)
                g_loss = g_loss_fn(fake_outputs, real_labels)

                # Update weights
                g_loss.backward()
                g_optimizer.step()

                #Progress tracking
                g_loss_val = g_loss.item()  # Assuming you calculated loss
                pbar.set_postfix({"Generator loss": g_loss_val})
                g_losses.append(g_loss.item())


                # Print progress
            

        return [{"generator": g_losses}, {"discriminator": d_losses}]
    except KeyboardInterrupt:
        print("Training interrupted. Cleaning up...")



if __name__ == "__main__":

    make_dir()

    dataloader= get_train_and_val(config.TRAIN_IMAGE_DATASET_PATH, config.TRAIN_MASK_DATASET_PATH, config.TRAIN_REMOVED_DATASET_PATH, config.BATCH_SIZE, config.IMAGE_SIZE, num_workers=config.NUM_WORKERS)

    device = config.DEVICE

    generator = ShadowGenerator().to(device)
    discriminator = ShadowDiscriminator().to(device)

    init_weights(generator)
    init_weights(discriminator)

    if torch.cuda.is_available():
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)
        print("CUDA enabled")

        g_optimizer = Adam(generator.parameters(), lr=config.INIT_LR, betas=(0.5, 0.999), weight_decay=1e-5)
        d_optimizer = Adam(discriminator.parameters(), lr=config.INIT_LR, betas=(0.5, 0.999), weight_decay=1e-5)

        losses = train_gan(generator, discriminator, dataloader, g_optimizer, d_optimizer, BCEWithLogitsLoss(), BCEWithLogitsLoss(), config.NUM_EPOCHS, device)

        plot_losses(losses, 'GAN losses')

        torch.save(generator.state_dict(), config.MODEL_OUTPUT_DIR + "/generator.pth")
        torch.save(discriminator.state_dict(), config.MODEL_OUTPUT_DIR + "/discriminator.pth")
        print("Models have been saved successfully.")