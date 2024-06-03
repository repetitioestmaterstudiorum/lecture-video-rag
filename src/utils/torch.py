import torch

# ---


def detect_gpu():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple

    print(f"GPU detected: {device}")
    return device
