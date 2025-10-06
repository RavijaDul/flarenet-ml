import os
import gdown

# Google Drive share link
DRIVE_URL = "https://drive.google.com/file/d/1qm5UqNOLb9KQtwF6HaxIxng1SFmYR3r7/view?usp=sharing"

# 🔹 Destination folder and file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEST_DIR = os.path.join(BASE_DIR, "model_weights")
DEST_PATH = os.path.join(DEST_DIR, "patchcore_model.pkl")

def download_model():
    # Create model_weights directory if it doesn't exist
    os.makedirs(DEST_DIR, exist_ok=True)

    print("🔹 Checking for existing model file...")
    if os.path.exists(DEST_PATH):
        print(f"✅ Model already exists at: {DEST_PATH}")
        return

    print("⬇️ Downloading patchcore_model.pkl from Google Drive...")
    try:
        # Use gdown to download the file from Drive
        gdown.download(DRIVE_URL, DEST_PATH, quiet=False, fuzzy=True)
        print(f"✅ Download complete! Model saved to: {DEST_PATH}")
    except Exception as e:
        print(f"❌ Download failed: {e}")

if __name__ == "__main__":
    download_model()
