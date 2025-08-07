import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Check CUDA version in PyTorch
print("CUDA version in PyTorch:", torch.version.cuda)

# If CUDA is available, get device information
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
