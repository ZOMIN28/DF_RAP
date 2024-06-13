import numpy as np
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    if x.min() < 0:
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    return x


def Image2tensor(imagepath,process=False,resize=256):
    img = Image.open(imagepath).convert("RGB")
    transform = []
    transform.append(T.ToTensor())
    if len(img.split()) == 3:
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    else:
       transform.append(T.Normalize(mean=0.5, std=0.5))
    if process:
        transform.append(T.Resize([resize,resize]))
    transform = T.Compose(transform)
    img = torch.unsqueeze(transform(img),dim=0).to(device)
    return img

    
