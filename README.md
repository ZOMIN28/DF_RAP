# DF-RAP: A Robust Adversarial Perturbation for Defending against Deepfakes in Real-world Social Network Scenarios
Implementation of "DF-RAP: A Robust Adversarial Perturbation for Defending against Deepfakes in Real-world Social Network Scenarios".
<img src="images\Real-world Scenarios.png" alt="Real-world Scenarios" style="zoom:67%;" />

### 1. Usage
Install the required dependency packages given in requirements.txt.

Follow Parts 2 and 3 to download and configure the pre-trained model and dataset.

You can follow `demo.ipynb` to implement the proposed DF-RAP.

### 2. Pretrained model 

The pretrained model of `ComGAN` and `PertG` is available in [ComGAN & PertG](https://drive.google.com/file/d/18opqlLzn5MCTboKkwcq58sSdkxKE3WOU/view?usp=drive_link). Put them in `DF-RAP/checkpoints/`.

The pretrained model of `SimSwap` and `Arcface` is available in [SimSwap (old)](https://drive.google.com/drive/folders/1tGqLa87UogpMoDbzthsclIcL52-jHbk_?usp=drive_link). Put them in `DF-RAP/SimSwap/arcface_model/`  and  `DF-RAP/SimSwap/checkpoints/`.

The pretrained model of `StarGAN` is available in [StarGAN](https://www.dropbox.com/s/zdq6roqf63m0v5f/celeba-256x256-5attrs.zip?dl=0). Put it in `DF-RAP/checkpoints/stargan_celeba_256/models/`.


### 3. Dataset
We have made the OSN transmission image dataset mentioned in this work publicly available. You can get it in [OSN-transmission mini CelebA](https://github.com/ZOMIN28/OSN-transmission_mini_CelebA), and put it in `DF-RAP/data/`.

### 4. Test: simple DF-RAP generation
You can generate the proposed DF-RAP using the following code:
```python
ComG_model = torch.load('checkpoints/ComGAN/ComG_model.pt')
ComG = ComG_model['ComG'].to(device)
ComG.eval()
starG = stargan_model()

adv_image = adv_attack(ori_image, c_trg=c_trg, model=starG, device=device, faketype="StarGAN", ComG=ComG, comgan=True)
```

The figure below shows the defense effect of robust adversarial perturbations derived using PGD as the basic attack.
<img src="images\output.png" alt="output" style="zoom:67%;" />

### 5. Downstream task: training a DF-RAP generator using ComGAN
Beyond this paper, we further explore the possibility of combining the proposed method with generation-based adversarial attacks. You can train a DF-RAP generator against StarGAN by running the:
```
python train_pG.py
```
After training, you can quickly generate DF-RAP using the following code:
```python
pertG = torch.load('checkpoints/PertG/df-rap_Gen_stargan.pt').to(device)
pertG.eval()

epsilon = 0.05
pert = pertG(ori_image)
pert = torch.clamp(pert, -epsilon, epsilon)
adv_image = torch.clamp(ori_image+pert, -1.0, 1.0)
```

The figure below shows the defense effect, and you can test it in demo. This shows that our work can be used as a plug-and-play plugin in the community.
<img src="images\output2.png" alt="output2" style="zoom:67%;" />

### 6. Citation
If our paper helps your research, please cite it in your publications:
```
@article{qu2024df,
  title={DF-RAP: A Robust Adversarial Perturbation for Defending against Deepfakes in Real-world Social Network Scenarios},
  author={Qu, Zuomin and Xi, Zuping and Lu, Wei and Luo, Xiangyang and Wang, Qian and Li, Bin},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024},
  publisher={IEEE}
}
```
### 7. License
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/?spm=a2c22.12281978.0.0.3e4732ffcOYDZo) (Attribution-NonCommercial-ShareAlike 4.0 International).

The code is released for academic research use only. For other use, please contact me.

### Acknowledges

Our work is based on:

[1] [DiffJPEG](https://github.com/mlomnitz/DiffJPEG)

[2] [StarGAN](https://github.com/yunjey/stargan)

[3] [SimSwap](https://github.com/neuralchen/SimSwap)

[4] [Disrupting](https://github.com/natanielruiz/disrupting-deepfakes)
