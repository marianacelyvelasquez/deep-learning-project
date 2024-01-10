import torch
import os


def get_device(device_type=None):
    if device_type is not None:
        if device_type == "cuda":
            n = torch.cuda.device_count()
            print(f"Using {n} CUDA devices")
            return "cuda", n
        elif device_type == "cpu":
            n = os.cpu_count()
            print("Using {n} CPU")
            return "cpu", n
        elif device_type == "mps":
            print("Using METAL GPU")
            return "mps", 1
        else:
            raise ValueError(f"Unknown device type {device_type}")

    # Check if CUDA is available
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"Using {n} CUDA devices")
        return "cuda", n
    # Check if MPS is available
    elif torch.backends.mps.is_available():
        print("Using METAL GPU")
        return "mps", 1
    # If neither CUDA nor MPS is available, use CPU
    else:
        n = os.cpu_count()
        print("Using {n} CPU")
        return "cpu", n
