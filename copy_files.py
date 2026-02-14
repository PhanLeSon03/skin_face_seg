import shutil
import os

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

MASK_DIR = '/home/jupyter/sonpl/data_wrinkle/manual_wrinkle_masks/'
TARGET_DIR = '/home/jupyter/sonpl/data_wrinkle/manual_wrinkle_images/'
SOURCE_DIR = '/home/jupyter/sonpl/data_wrinkle/ffhq-dataset/images1024x1024/'
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

os.makedirs(TARGET_DIR, exist_ok=True)

mask_images = [f for f in os.listdir(MASK_DIR) if os.path.isfile(os.path.join(MASK_DIR, f)) and f.lower().endswith(VALID_EXTENSIONS)]

for mask_image in mask_images:
    filename = mask_image.split(".")[0]
    int_filename = int(filename)
    folder_number = (int_filename // 1000) * 1000
    folder_name = f"{folder_number:05d}" 

    source_path = os.path.join(SOURCE_DIR, folder_name, mask_image)
    target_path = os.path.join(TARGET_DIR, mask_image)

    print(f"Copying from {source_path} to {target_path}")
    
    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)
    else:
        print(f"File not found: {source_path}")

    
    
