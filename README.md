# DF-RAP: A Robust Adversarial Perturbation for Defending against Deepfakes in Real-world Social Network Scenarios
Implementation of "DF-RAP: A Robust Adversarial Perturbation for Defending against Deepfakes in Real-world Social Network Scenarios".
<img src="images\Real-world Scenarios.png" alt="Real-world Scenarios" style="zoom:67%;" />

### 1、Usage
Install the required dependency packages given in requirements.txt.

You can follow `demo.ipynb` to implement robust adversarial attacks against Deepfakes.

### 2、Pretrained model

The pretrained model of `ComGAN` and `PertG` is available in [ComGAN & PertG](https://drive.google.com/file/d/1Hk-oraxtStH16BPf_2dveMdrncTSJOcI/view?usp=drive_link). Put them in `DF-RAP/checkpoints/`  .

The pretrained model of `SimSwap` and `Arcface` is available in [SimSwap (old)](https://drive.google.com/drive/folders/1tGqLa87UogpMoDbzthsclIcL52-jHbk_?usp=drive_link). Put them in `DF-RAP/SimSwap/arcface_model/`  and  `DF-RAP/SimSwap/checkpoints/` .

The pretrained model of `StarGAN` is available in [StarGAN](https://www.dropbox.com/s/zdq6roqf63m0v5f/celeba-256x256-5attrs.zip?dl=0). Put it in `DF-RAP/checkpoints/stargan_celeba_256/models/`.


### 3、Dataset
We have made the OSN transmission image dataset mentioned in this work publicly available. You can get it here [OSN-transmission mini CelebA](https://github.com/ZOMIN28/OSN-transmission_mini_CelebA).

### 4、Example
The figure below shows the defense effect of robust adversarial perturbations derived using PGD as the basic attack.
<img src="images\output.png" alt="output" style="zoom:67%;" />

### 5、Downstream tasks
Beyond this paper, we further explore the possibility of combining the proposed method with generation-based adversarial attacks. The implementation details are given in the demo. The figure below shows the defense effect on StarGAN. This shows that our work can be used as a plug-and-play plugin in the community.
<img src="images\output2.png" alt="output2" style="zoom:67%;" />

### 6、Citation

```
@article{qu2024df,
  title={DF-RAP: A Robust Adversarial Perturbation for Defending against Deepfakes in Real-world Social Network Scenarios},
  author={Qu, Zuomin and Xi, Zuping and Lu, Wei and Luo, Xiangyang and Wang, Qian and Li, Bin},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024},
  publisher={IEEE}
}
```

### Acknowledges

Our work is based on:

[1] [DiffJPEG](https://github.com/mlomnitz/DiffJPEG)

[2] [StarGAN](https://github.com/yunjey/stargan)

[3] [SimSwap](https://github.com/neuralchen/SimSwap)

[4] [Disrupting](https://github.com/natanielruiz/disrupting-deepfakes)
