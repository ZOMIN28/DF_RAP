import torch
import net.stargan as stargan
from utils.utils import denorm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import sys
sys.path.insert(0, 'SimSwap/')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
"""
---------------------------------------------------------------
                            stargan
---------------------------------------------------------------                      
"""

def stargan_model(conv_dim=64,c_dim=5, repeat_num=6):
    starG = stargan.Generator(conv_dim=conv_dim, c_dim=c_dim, repeat_num=repeat_num)
    G_path = "checkpoints/stargan_celeba_256/models/200000-G.ckpt"
    starG.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    starG = starG.to(device)
    return starG


def stargan_fake(img, c_trg, starG):
    with torch.no_grad():
        gen_img = starG(img, c_trg)
    return gen_img


"""
---------------------------------------------------------------
                            simswap
---------------------------------------------------------------                      
"""
def simswap_model(opt):
    from SimSwap.models.models import create_model
    model = create_model(opt)
    model.eval()
    return model.to(device)

def processorg_simswap():
    transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([256,256])
    ])
    path = "data/simswap_target/target.png"
    img_a_list = [path]
    img_att_list = []
    with torch.no_grad():
        for pic_a in img_a_list:
            img_a = Image.open(pic_a).convert('RGB')
            img_a = transformer_Arcface(img_a)
            img_att = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).to(device)
            img_att_list.append(img_att)
    return img_att_list

def simswap_fake(img_att, img_id, simG):
    with torch.no_grad():
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = simG.netArc(img_id_downsample)
        latend_id = latend_id / torch.norm(latend_id, p=2, dim=1, keepdim=True)
        img_fake = simG(img_id, img_att, latend_id, latend_id, True)
    return img_fake