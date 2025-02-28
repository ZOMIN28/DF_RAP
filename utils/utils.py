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


def create_labels(c_org, c_dim=5, selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
            hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        c_trg = c_org.clone()
        if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
            c_trg[:, i] = 1
            for j in hair_color_indices:
                if j != i:
                    c_trg[:, j] = 0
        else:
            c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
        c_trg_list.append(c_trg.to(device))
    return c_trg_list
