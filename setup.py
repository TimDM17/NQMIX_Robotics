# Install required packages
# !pip install gymnasium-robotics[mujoco-py]
# !pip install torch torchvision

# Check if GPU is available
import torch
print("="*70)
print("GPU Check:")
print("="*70)

if torch.cuda.is_available():
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print(f"No GPU detected - training will use CPU (slower)")

print("=" * 70)