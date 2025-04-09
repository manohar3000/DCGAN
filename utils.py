import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from config import SAVED_IMAGES_DIR

def save_and_plot_images(generator, fixed_noise, iteration, device):
    """
    Generate and save images from fixed noise using the generator.
    """
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()
    
    img_grid = make_grid(fake_images, padding=2, normalize=True)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(img_grid, (1, 2, 0)))
    plt.axis('off')

    os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)
    save_path = os.path.join(SAVED_IMAGES_DIR, f"image_{iteration}.png")
    plt.savefig(save_path, bbox_inches='tight')
    
    plt.close()
    generator.train()
