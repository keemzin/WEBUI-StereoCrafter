#!/bin/bash
set -e

echo "=========================================="
echo "StereoCrafter RunPod Setup Script"
echo "=========================================="
echo "Running from: $(pwd)"
echo ""

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Private model downloads may fail."
    echo "Set HF_TOKEN in RunPod environment variables"
fi

# Step 1: Clone original StereoCrafter to temp location
echo ""
echo "Step 1: Cloning original StereoCrafter repository to temp..."
cd ..
git clone https://github.com/enoky/StereoCrafter.git StereoCrafter-temp
cd StereoCrafter-temp
git checkout 2a1d473
cd ..

# Step 2: Copy missing files from original to current directory (skip existing)
echo ""
echo "Step 2: Merging original StereoCrafter files (preserving your patch files)..."
cp -rn StereoCrafter-temp/* WEBUI-StereoCrafter/
echo "Merge complete! Your patch files were preserved."

# Step 3: Clean up temp directory
echo ""
echo "Step 3: Cleaning up temporary files..."
rm -rf StereoCrafter-temp
cd WEBUI-StereoCrafter

# Step 4: Fix requirements.txt for Linux (remove Windows-specific packages)
echo ""
echo "Step 4: Fixing requirements.txt for Linux..."
sed -i '/triton-windows/d' requirements.txt
sed -i '/ttkthemes/d' requirements.txt
echo "Removed Windows-specific packages"

# Step 5: Create weights folder and download models
echo ""
echo "Step 5: Creating weights folder and downloading models..."

# Install huggingface_hub first (needed for authentication)
echo "Installing huggingface_hub..."
pip install huggingface_hub

mkdir -p weights
cd weights

# Download stable-video-diffusion (requires HF token)
echo "Downloading stable-video-diffusion-img2vid-xt-1-1..."
if [ -n "$HF_TOKEN" ]; then
    # Configure git credential helper first
    echo "Configuring git credential helper..."
    git config --global credential.helper store
    
    # Use Python's huggingface_hub to login
    echo "Logging into Hugging Face..."
    python -c "from huggingface_hub import login; login(token=\"$HF_TOKEN\", add_to_git_credential=True)"
    
    # Verify login worked
    if [ $? -eq 0 ]; then
        echo "HuggingFace login successful"
        # Clone using standard URL (credentials are now stored)
        git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
    else
        echo "ERROR: HuggingFace login failed"
        exit 1
    fi
else
    echo "ERROR: HF_TOKEN environment variable not set"
    echo "Please set HF_TOKEN in RunPod environment variables"
    exit 1
fi

# Download DepthCrafter
echo "Downloading DepthCrafter..."
git clone https://huggingface.co/tencent/DepthCrafter

# Download StereoCrafter weights
echo "Downloading StereoCrafter weights..."
git clone https://huggingface.co/TencentARC/StereoCrafter

cd ..

# Step 6: Install system dependencies (fix tkinter error)
echo ""
echo "Step 6: Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq python3-tk

# Step 7: Install Python dependencies
echo ""
echo "Step 7: Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Working directory: $(pwd)"
echo ""
echo "IMPORTANT: Set this environment variable before running:"
echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""
echo "To UPDATE your patch in the future, just run:"
echo "  git pull origin main"
echo ""
echo "To start the application:"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  python webui.py --share --server-name 0.0.0.0 --server-port 7860"
echo ""
