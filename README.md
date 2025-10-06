# flarenet-ml
# Electrical Component Anomaly Detection

Thermal imaging anomaly detection for electrical components using Patchcore model.

## Project Structure
```
Model_Inference/
├── load_model.py           # Load model once (run first)
├── run_inference.py        # Fast inference (recommended)  
├── config/
│   └── patchcore_transformers.yaml
├── model_weights/
│   ├── model.ckpt          # Original checkpoint
│   └── patchcore_model.pkl # Pre-loaded model
├── test_image/             # Input images
├── labeled_segmented/      # Segmented anomaly outputs
└── output_image/           # Final labeled images
```

## Quick Setup
```powershell
# Install dependencies
pip install torch opencv-python pillow numpy omegaconf anomalib

# Verify installation  
python -c "import torch, cv2, anomalib; print('✅ Ready')"
```

## Usage

### Fast Method (Recommended)
```powershell

# activate the virtual environment 
python -m venv venv

venv\Scripts\activate   # Windows:

pip install -r requirements.txt

# Download the model weights
python model_weights.py

# 1. Load model once
python load_model.py

# 2. Run inference (repeat for each image)
python run_inference.py
```

### Single Command (Slower)
```powershell
python pipeline.py
```

## Add Your Images
1. Place images in `test_image/` folder
2. Edit image path in script:
   ```python
   TEST_IMAGE = os.path.join(BASE_DIR, "test_image", "your_image.jpg")
   ```
3. Run inference

## Detection Types
| Type | Box Color | Description |
|------|-----------|-------------|
| Normal | Yellow text | No issues |
| Point Overload (Faulty) | Red | Critical hot spots |
| Point Overload (Potential) | Yellow | Warning areas |
| Loose Joint (Faulty) | Red | Critical connections |
| Loose Joint (Potential) | Yellow | Potential issues |
| Full Wire Overload | Yellow | Entire wire heating |

## Commands
```powershell
# Check outputs
dir labeled_segmented\  # Segmented images
dir output_image\       # Final images

# Process different image
# Edit TEST_IMAGE path in script, then run
```

## Troubleshooting
```powershell
# Missing anomalib
pip install anomalib

# Model not found  
python load_model.py

# CUDA memory issues
set CUDA_VISIBLE_DEVICES=""
```