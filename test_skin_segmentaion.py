import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
from skimage.filters import frangi, gabor
from skimage import measure, morphology
import pathlib
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Skin_metrics/wrinkle/')))
import model 


# DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TRANSFORM (Resize + Tensor)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
Script_Dir = str(pathlib.Path(__file__).parent.absolute())


model = Script_Dir + 'checkpoints/model_epoch_20.pth'
model = striped_attn.StripedWriNet(n_channels=3, n_classes=2).to(DEVICE)
checkpoint1 = torch.load(model, map_location=DEVICE)
model.load_state_dict(checkpoint1['model_state_dict'])
model.eval()


def predict(image: Image.Image, model, transform, device):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        mask_np = predicted.squeeze().cpu().numpy()
        return Image.fromarray((mask_np * 255).astype('uint8'))


def apply_mask(image: Image.Image, mask: Image.Image):
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    image_np = np.array(image)
    mask_np = np.array(mask)
    if len(mask_np.shape) == 2:
        mask_np = mask_np[:, :, np.newaxis]
    binary_mask = (mask_np > 0).astype(np.uint8)
    masked_np = image_np * binary_mask
    return Image.fromarray(masked_np.astype('uint8'))

def overlay_mask_on_image(input_image, predicted_mask, alpha=0.5, wrinkle_color=(255, 0, 0)):
    # Resize the input image to match the mask size (256x256)
    input_image = input_image.resize((256, 256), Image.Resampling.LANCZOS)

    # Convert the predicted mask to a numpy array
    mask_np = np.array(predicted_mask)

    # Create a colored mask (e.g., red for wrinkles)
    colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    colored_mask[mask_np == 255] = wrinkle_color  

    # Convert the colored mask to a PIL image
    colored_mask_image = Image.fromarray(colored_mask)

    # Convert the input image to RGBA for blending
    input_image = input_image.convert('RGBA')
    colored_mask_image = colored_mask_image.convert('RGBA')

    # Create a transparency mask: where the mask is 255, use alpha; otherwise, fully transparent
    mask_alpha = np.zeros_like(mask_np, dtype=np.uint8)
    mask_alpha[mask_np == 255] = int(255 * alpha)  # Apply transparency to wrinkle areas
    mask_alpha_image = Image.fromarray(mask_alpha, mode='L')

    # Blend the input image and colored mask
    overlay = Image.composite(colored_mask_image, input_image, mask_alpha_image)

    return overlay

def predict_mask(input_image):
    skin_mask = predict(input_image, model, transform, DEVICE)
    masked_input = apply_mask(input_image, skin_mask)
    #masked_input.save(f'masked_input.png')
 
    overlay_image = overlay_mask_on_image(input_image, skin_mask, alpha=0.5, wrinkle_color=(0, 0, 255))

    return overlay_image, masked_input


if __name__ == "__main__":
    filename_image = 'image_old_lady.png'
    os.makedirs("images", exist_ok=True)
    
    input_path = f'images/{filename_image}'
    filename = filename_image.split('.')[0]

    input_image = Image.open(input_path).convert('RGB')

    mask_256, masked_input = predict_mask(input_image) 

    width, height = input_image.size

    wrinkle_image = mask_256.resize((width, height), Image.Resampling.LANCZOS)

    # Save
    wrinkle_image.save(f'images/{filename}_mask.png')

    print("Done.")
