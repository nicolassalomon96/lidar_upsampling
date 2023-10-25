import torch
from swd_all import *

# Example usage:
if __name__ == "__main__":

    batch = 1
    points = 65536
    channels = 3
    x1 = torch.rand((batch, points, channels), requires_grad=True)  # 1024 images, 3 chs, 128x128 resolution
    x2 = torch.rand((batch, points, channels), requires_grad=True)
    out = swd(x1, x2, device="cuda") # Fast estimation if device="cuda"
    print(out) # tensor(53.6950)