from huggingface_hub import snapshot_download
import os

# Folder where you want to save the dataset
save_dir = "/mnt/swordfish-pool2/kavin/benchmarking/architecture-annotations"

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Download the dataset
snapshot_download(
    repo_id="kr3131/architecture-annotations",
    repo_type="dataset",
    local_dir=save_dir,
    local_dir_use_symlinks=False  # ensures full download (no symlinks)
)

print("Download completed! Saved to:", save_dir)