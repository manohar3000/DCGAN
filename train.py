import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import DEVICE, DATASET_PATH, BATCH_SIZE, NUM_WORKERS, Z_DIM, LEARNING_RATE, BETAS, NUM_EPOCHS, FIXED_NOISE_SIZE, SAVED_IMAGES_DIR, SAVE_IMAGE_INTERVAL
from dataset import CustomDataset
from models import Generator, Discriminator
from utils import save_and_plot_images

def main():
    # Create dataset and dataloader
    dataset = CustomDataset(root_dir=DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    # Initialize models
    generator = Generator(z_dim=Z_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    # Loss function: Binary Cross Entropy
    criterion = nn.BCELoss()
    
    # Optimizers for both Generator and Discriminator
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    
    # Fixed noise for evaluating Generator progress over training
    fixed_noise = torch.randn(FIXED_NOISE_SIZE, Z_DIM, 1, 1, device=DEVICE)
    
    # Lists to store loss values
    G_losses = []
    D_losses = []
    iteration = 0

    # Create directory for saved images if needed
    os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)
    
    print("Starting Training Loop...")
    for epoch in range(NUM_EPOCHS):
        for i, real_images in enumerate(dataloader, 0):
            # ----- Update Discriminator -----
            discriminator.zero_grad()
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)
            label_real = torch.ones(batch_size, device=DEVICE)
            output_real = discriminator(real_images).view(-1)
            errD_real = criterion(output_real, label_real)
    
            noise = torch.randn(batch_size, Z_DIM, 1, 1, device=DEVICE)
            fake_images = generator(noise)
            label_fake = torch.zeros(batch_size, device=DEVICE)
            output_fake = discriminator(fake_images.detach()).view(-1)
            errD_fake = criterion(output_fake, label_fake)
    
            errD = errD_real + errD_fake
            errD.backward()
            optimizer_d.step()
            D_losses.append(errD.item())
    
            # ----- Update Generator -----
            generator.zero_grad()
            # Generator tries to trick the discriminator into classifying fake images as real
            label_gen = torch.ones(batch_size, device=DEVICE)
            output_gen = discriminator(fake_images).view(-1)
            errG = criterion(output_gen, label_gen)
            errG.backward()
            optimizer_g.step()
            G_losses.append(errG.item())
    
            iteration += 1
    
            # Print progress at regular intervals
            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{NUM_EPOCHS}][Batch {i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")
    
            # Save and plot images every SAVE_IMAGE_INTERVAL iterations
            if iteration % SAVE_IMAGE_INTERVAL == 0:
                save_and_plot_images(generator, fixed_noise, iteration, DEVICE)

if __name__ == "__main__":
    main()
