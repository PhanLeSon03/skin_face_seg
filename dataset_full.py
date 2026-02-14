import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

class FFHQWrinkleDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = []

        # Traverse subfolders in image_dir to collect all image files
        for subfolder in sorted(os.listdir(image_dir)):
            subfolder_path = os.path.join(image_dir, subfolder)
            if os.path.isdir(subfolder_path):  # Ensure it's a directory
                if int(subfolder) < 50000:
                    for img_name in sorted(os.listdir(subfolder_path)):
                        if os.path.isfile(os.path.join(subfolder_path, img_name)) and img_name.lower().endswith(VALID_EXTENSIONS):
                            # Store the subfolder and image name as a tuple
                            self.images.append((subfolder, img_name))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the subfolder and image name
        subfolder, img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, subfolder, img_name)
        mask_path = os.path.join(self.mask_dir, subfolder,img_name)  

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Normalize mask to 0 or 1
        mask = torch.where(mask > 0, torch.tensor(1.0), torch.tensor(0.0))

        return image, mask

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match StripedWriNet input
    transforms.ToTensor(),
])