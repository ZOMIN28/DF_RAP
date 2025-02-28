import numpy as np
import torch
from utils.utils import denorm
import torch.nn.functional as Func
import torch
from deepfakes import stargan_fake, simswap_fake

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, ComG=None, ComG_woj=None, epsilon=0.05, k=10, a=0.01, balance=1.0):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.balance = balance
        self.loss_fn = Func.mse_loss
        self.device = device
        self.ComG = ComG
        self.ComG_woj = ComG_woj
        # PGD or I-FGSM?
        self.rand = True

    def perturb(self, X_nat, y, c_trg=None, img_att=None,latend_id=None, faketype=None, comgan=True):

        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
        
        
        for i in range(self.k):
            X.requires_grad = True

            if faketype == "StarGAN":
                self.model.zero_grad()
                if not comgan:
                    output, _ = self.model.features(X, c_trg)
                else:
                    self.ComG.zero_grad()
                    output1, _ = self.model.features(self.ComG(X), c_trg) 

                    if self.ComG_woj is not None:
                        self.ComG_woj.zero_grad()
                        output2,_ = self.model.features(self.ComG_woj(X), c_trg)
                        # You can adjust parameters to balance robust defenses against different deepfake models
                        output = self.balance*output1 + (1.0-self.balance)*output2
                    else:
                        output = output1

            elif faketype == "simswap":
                self.model.zero_grad()
                if not comgan:
                    img_id_downsample = Func.interpolate(X, size=(112,112))
                    latend_id = self.model.netArc(img_id_downsample)
                    latend_id = latend_id / torch.norm(latend_id, p=2, dim=1, keepdim=True)
                    output = self.model(X, img_att, latend_id, latend_id, True)
                else:
                    self.ComG.zero_grad()
                    img_id_downsample1 = Func.interpolate(self.ComG(X), size=(112,112))
                    latend_id1 = self.model.netArc(img_id_downsample1)
                    latend_id1 = latend_id1 / torch.norm(latend_id1, p=2, dim=1, keepdim=True)
                    output1 = self.model(self.ComG(X), img_att, latend_id1, latend_id1, True)

                    if self.ComG_woj is not None:
                        self.ComG_woj.zero_grad()
                        img_id_downsample2 = Func.interpolate(self.ComG_woj(X), size=(112,112))
                        latend_id2 = self.model.netArc(img_id_downsample2)
                        latend_id2 = latend_id2 / torch.norm(latend_id2, p=2, dim=1, keepdim=True)
                        output2 = self.model(self.ComG_woj(X), img_att, latend_id2, latend_id2, True)
                        # You can adjust parameters to balance robust defenses against different deepfake models
                        output = self.balance*output1 + (1.0-self.balance)*output2
                    else:
                        output = output1

            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, y)
            loss.backward()

            grad = X.grad
            X_adv = X + self.a * grad.sign()
            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()
        self.model.zero_grad()
        return X, X - X_nat





def adv_attack(img, ComG, ComG_woj, model=None,device=None, epsilon=0.05, c_trg=None, img_att=None, faketype="starGAN",comgan=True,balance=1.0):
    X = None
    pgd_attack = LinfPGDAttack(model=model, device=device, epsilon=epsilon, ComG=ComG, ComG_woj=ComG_woj, balance=balance)

    if faketype == "StarGAN":
        with torch.no_grad():
            x_real = img
            gen_noattack = stargan_fake(x_real,c_trg,model)
        _,perturb = pgd_attack.perturb(X_nat=img,y=gen_noattack,c_trg=c_trg,faketype=faketype,comgan=comgan)
        X = img + perturb

    elif faketype == "simswap":
        with torch.no_grad():
            x_real = img
            gen_noattack = simswap_fake(img_att,x_real,model)
        _,perturb = pgd_attack.perturb(X_nat=img,y=gen_noattack,img_att=img_att,faketype=faketype,comgan=comgan)
        X = img + perturb


    return X
    
