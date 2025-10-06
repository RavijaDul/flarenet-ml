import os
import torch
from omegaconf import OmegaConf
from anomalib.models import Patchcore
import pickle  # to serialize the model

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "patchcore_transformers.yaml")
CKPT_PATH = os.path.join(BASE_DIR, "model_weights", "model.ckpt")
MODEL_FILE = os.path.join(BASE_DIR, "model_weights", "patchcore_model.pkl")

# -------------------------
# Load Patchcore model
# -------------------------
print("🔹 Loading Patchcore model...")
config = OmegaConf.load(CONFIG_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Patchcore.load_from_checkpoint(CKPT_PATH, **config.model.init_args)
model.eval()
model = model.to(device)
print("✅ Model loaded.")

# -------------------------
# Save model for later inference
# -------------------------
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved at {MODEL_FILE}")
