import torch

def get_device(self):
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    # Check if MPS is available
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using METAL GPU")
    # If neither CUDA nor MPS is available, use CPU
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device