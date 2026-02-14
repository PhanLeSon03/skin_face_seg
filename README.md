# Skin face Segmentation 

---
## Dataset
Manual and auto labeling for wrinle:
https://github.com/labhai/ffhq-wrinkle-dataset/tree/main

Download full images:
https://github.com/NVlabs/ffhq-dataset


gdown https://drive.google.com/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA #json file
python download_ffhq.py --images

## Training Process


- **Goal**: Train on a large-scale **auto-labeled** dataset (~50K images).
- **Script**: `train.py`

Epoch 20/20: 100%|███████████████████████████████████████████████████████████████████| 3125/3125 [1:06:15<00:00,  1.27s/it]
Epoch [20/20], Loss: 0.0157

Epoch 20/20: 100%|████████████████████████████████████████████████████████████████████████████████████| 3125/3125 [41:10<00:00,  1.26it/s]
Epoch [20/20], Loss: 0.0145
Saved checkpoint: checkpoints/model_epoch_20.pth


## File Descriptions

| File | Description |
|------|------------|
| `copy_files.py` | Copies manually labeled images from the sub-folder structure `images1024x1024/` into a flat folder `manual_wrinkle_images/`. |
| `dataset_full.py` | Dataset loader that supports loading data from a hierarchical sub-folder structure. |
| `train.py` | Training script for Stage 1 using the auto-labeled dataset. |
| `test_skin_segmentaion.py` | Example script for running inference. |
| `striped_wrinet.py` | Implementation of the StripedWriNet model architecture. |

---
                                                                        |
## Dataset
data_wrinkle/
├── images1024x1024/              # ~70K original facial images in sub-folder structure
├── weak_wrinkle_masks/           # ~50K auto-labeled wrinkle masks (Stage 1)



