import config
from model import ShadowGenerator,ShadowDiscriminator,init_weights
from dataset import get_train_and_val
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



def plot_losses(losses_G, losses_D):
    plt.plot(losses_G, label='Generator Loss')
    plt.plot(losses_D, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
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

def train(netG, netD, train_loader, val_loader, device, num_epochs, lr):
    #set models to training mode
    netG.train()
    netD.train()
    
    torch.backends.cudnn.benchmark = True

    #initialize optimizers and loss function
    optimizerG = Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = BCEWithLogitsLoss()

    losses_G = []
    losses_D = []

    for epoch in range(num_epochs):
        for images, masks in tqdm(train_loader):

            images = images.to(device)
            masks = masks.to(device)

            # Train the Discriminator
            netD.zero_grad()
            real = torch.cat((images, masks), 1)
            output = netD(real)
            lossD_real = criterion(output, torch.ones_like(output))
            lossD_real.backward()

            fake_masks = netG(images)
            fake = torch.cat((images, fake_masks), 1)
            output = netD(fake.detach())
            lossD_fake = criterion(output, torch.zeros_like(output))
            lossD_fake.backward()

            lossD = lossD_real + lossD_fake
            optimizerD.step()

            losses_D.append(lossD.item())
        
            # Train the Generator
            netG.zero_grad()
            fake = torch.cat((images, fake_masks), 1)
            output = netD(fake)
            lossG = criterion(output, torch.ones_like(output))
            lossG.backward()
            optimizerG.step()

            losses_G.append(lossG.item())

        print(f"Epoch {epoch+1}/{num_epochs} Loss G: {losses_G[epoch]} Loss D: {losses_D[epoch]}")
    return losses_G, losses_D           


            

if __name__ == "__main__":

    make_dir()

    train_loader, val_loader = get_train_and_val(config.TRAIN_IMAGE_DATASET_PATH, config.TRAIN_MASK_DATASET_PATH, config.BATCH_SIZE, config.IMAGE_SIZE, num_workers=config.NUM_WORKERS)

    device = config.DEVICE

    generator = ShadowGenerator().to(device)
    discriminator = ShadowDiscriminator().to(device)

    init_weights(generator)
    init_weights(discriminator)

    if device == 'cuda':
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)


    g_losses, d_losses = train(generator, discriminator, train_loader, val_loader, device, config.NUM_EPOCHS, config.INIT_LR)

    plot_losses(g_losses, d_losses)

    torch.save(generator.state_dict(), config.MODEL_OUTPUT_DIR + "/generator.pth")
    torch.save(discriminator.state_dict(), config.MODEL_OUTPUT_DIR + "/discriminator.pth")
    print("Models have been saved successfully.")
