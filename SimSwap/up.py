import torch
import sys
sys.path.append("..")
import torchvision
from utils.data_loader import get_loader
from utils.FIAloss import FIAloss
from net.mygan import Generator
from torchvision import transforms as T
import numpy as np
from torchvision.utils import save_image,make_grid
from torchvision.transforms import Resize
from utils.pytorch_msssim import ssim
from tqdm import tqdm
from deepfakes import stargan_model, stargan_attack, attgan_attack, attgan_model, attentiongan_model, attentiongan_attack,simswap_model,simswap_attack
import torch.nn.functional as F
import os
from options.test_options import TestOptions
import attacks


def getDataloader(type,image_size,batch_size):
    if type == "simswap":
        celeba_image_dir = '../data/celeba-256/images'
        attr_path = '../data/celeba-256/list.txt'
        selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        celeba_loader = get_loader(celeba_image_dir, attr_path, selected_attrs, image_size=image_size, batch_size=batch_size, mode="test", num_workers=0)
    return celeba_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
"""参数设置"""
image_size = 256
batch_size = 1
mask_prob = 0.8
mask_num = 30
epochs = 30  # (100)
alpha = 0.01 # (0.08)
epsilon = 0.05
mu = 0.05 #(0.5)
name = [str(i) for i in range(0,100)] 






def disrupting_p(epsilon=0.05,type="starGAN"):
    opt = TestOptions().parse()
    celeba_loader = getDataloader(type,image_size,batch_size)
    simG = simswap_model(opt)

    loss_sum = 0
    count = 0
    ssim_sum = 0
    for n,(img,c_org) in enumerate(tqdm(celeba_loader)):
        img = img.to(device)
        c_org = c_org.to(device)
        if type == "simswap":
            # pgd_attack = attacks.LinfPGDAttack(model=starG, device=device, epsilon=epsilon,feat=None)
            # with torch.no_grad():
            #     x_real = img
            #     gen_noattack = starG(x_real,c_org)
            # x_adv,perturb = pgd_attack.perturb(img, gen_noattack,c_org)
            # X = img + perturb
            #save_image(denorm(X[0].data.cpu()),"save/adv_img/"+name[n]+".png")
            gen_list_cle = simswap_attack(img, simG, name[n]+"_cle",save=True)
           # gen_list_adv = simswap_attack(X, simG, name[0]+"_adv",save=True)
    #         loss = 0
    #         Ssim = 0
    #         for i in range(len(gen_list_adv)):
    #             loss += F.mse_loss(gen_list_adv[i],gen_list_cle[i]) / 5
    #             Ssim += ssim(gen_list_adv[i],gen_list_cle[i], data_range=1.0, size_average=True) /5
    #             if(F.mse_loss(gen_list_adv[i],gen_list_cle[i]))>0.05:
    #                 count += 1
    #         loss_sum += loss.item()
    #         ssim_sum += Ssim.item()
    # print(loss_sum/100)
    # print(count/500)
    # print(ssim_sum/100)
        



# universal_p(epsilon=0.05,type="attGAN")
# universal_p(epsilon=0.05,type="starGAN")
# universal_p(epsilon=0.05,type="attentiongan")
#disrupting_p(epsilon=0.05,type="attGAN")
disrupting_p(epsilon=0.05,type="simswap")
# disrupting_p(epsilon=0.05,type="attentiongan")