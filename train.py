import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from dataset_full import FFHQWrinkleDataset, transform
import model

# Hyperparameters
BATCH_SIZE = 16  # Reduced due to high-resolution images
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 2  # Binary segmentation (background and face skin)

# Paths
IMAGE_DIR = '/home/jupyter/sonpl/data_wrinkle/ffhq-dataset/images1024x1024/'
MASK_DIR = '/home/jupyter/sonpl/data_wrinkle/weak_wrinkle_masks/'

CHECKPOINT_DIR = 'checkpoints/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize dataset and dataloader
dataset = FFHQWrinkleDataset(IMAGE_DIR, MASK_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

# Initialize model, loss, and optimizer
model = model.StripedWriNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()  # For binary segmentation
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).long()  # CrossEntropyLoss expects long labels

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks.squeeze(1))  # Remove channel dim from masks

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print epoch loss
    avg_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')

    # Save checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, checkpoint_path)
    print(f'Saved checkpoint: {checkpoint_path}')

print('Training finished!')
