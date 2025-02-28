import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--gpus", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--image-dir", type=str, default='data/OSN-transmission_mini_CelebA/original_images/')
parser.add_argument("--attr-path", type=str, default='data/OSN-transmission_mini_CelebA/attributes.txt')
parser.add_argument("--selected-attrs", type=list, default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
args = parser.parse_args()

import warnings
warnings.filterwarnings("ignore", category=Warning)
import torch
import numpy as np
import random
from net.PertG import PertGenerator
import torch.nn.functional as F
from tqdm import tqdm
from deepfakes import stargan_model, processorg_stargan
from utils.data_loader import get_loader
from utils.utils import denorm
from torchvision.utils import save_image
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


class AdvG_Attack:
    def __init__(self, device, dim_in=3, epsilon=0.05, deepfake=None):

        self.device = device
        self.dim_in = dim_in
        self.epsilon = epsilon
        self.lr = 0.0001
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.savelist = []
        self.count = 0

        self.loss_fn = F.mse_loss

        self.netG = PertGenerator(input_nc=dim_in).to(self.device)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), self.lr, [self.beta1, self.beta2])

        self.deepfake = deepfake.to(self.device)

        self.comgan = torch.load('checkpoints/ComGAN/ComG_model.pt')
        self.comgan = self.comgan['ComG'].to(device)
        self.comgan.eval()

    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def train_batch(self, x_real, c_org):
        savelist = []
        total_loss = 0.
        self.netG.train()
        perturbation = self.netG(x_real)
        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_real + perturbation, -1.0, 1.0)

        refer = processorg_stargan(c_org)
        r = refer[int(self.count%len(refer))]

        output_real = self.deepfake(x_real, r)
        output_adv = self.deepfake(self.comgan(x_adv), r)

        loss_G = -self.loss_fn(output_adv, output_real) 
        total_loss += loss_G
       
        self.optimizer_G.zero_grad()
        total_loss.backward()
        self.optimizer_G.step()
        self.count += 1

        if self.count % 10 == 0:
            for i in (x_real):
                savelist.append(i)
            for i in (output_real):
                savelist.append(i)
            for i in (output_adv):
                savelist.append(i)
            save_image(denorm(torch.stack(savelist).cpu()),"disrupting.png", nrow=args.batch_size, padding=0)

        return -loss_G.item()


    def train(self, train_dataloader, epochs = 2):
        for epoch in range(1, epochs+1): 
            loss_G = 0.
            with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
                for i,(img, c_org) in enumerate(train_dataloader):
                    img = img.to(self.device)
                    c_org = c_org.to(self.device)
                    loss_g = self.train_batch(img, c_org)
                    with torch.no_grad(): 
                        loss_G += loss_g
                    pbar.set_postfix(loss_G = loss_G / (i+1))
                    pbar.update()

            if epoch % 1 == 0:
                torch.save(self.netG, "checkpoints/df-rap_Gen_stargan.pt")


if __name__ == '__main__':
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    train_loader = get_loader(args.image_dir, args.attr_path, args.selected_attrs, 
                              image_size=256, batch_size=args.batch_size, shuffle=14,mode="train", num_workers=4)
    stargan = stargan_model()
    advG = AdvG_Attack(device=device, deepfake=stargan)
    advG.train(train_loader, 20)

                
