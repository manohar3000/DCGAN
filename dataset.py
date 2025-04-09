import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import DATASET_PATH, IMAGE_SIZE

class CustomDataset(Dataset):
    """
    Custom Dataset for loading images from a directory.

    Args:
        root_dir (str): Directory with all the images.
        transform (callable, optional): Transform to be applied on a sample.
    """
    def __init__(self, root_dir=DATASET_PATH, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # List all files in the directory
        self.images = os.listdir(root_dir)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Construct full image path
        img_name = os.path.join(self.root_dir, self.images[idx])
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_name}: {e}")
        
        if self.transform:
            image = self.transform(image)
        return image
