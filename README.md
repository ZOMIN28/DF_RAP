# DF-RAP: A Robust Adversarial Perturbation for Defending against Deepfakes in Real-world Social Network Scenarios
Implementation of "DF-RAP: A Robust Adversarial Perturbation for Defending against Deepfakes in Real-world Social Network Scenarios".
<img src="images\Real-world Scenarios.png" alt="Real-world Scenarios" style="zoom:67%;" />

### Usage

You can follow `demo.ipynb` to implement robust adversarial attacks against Deepfakes.

### Pretrained model

The pretrained model of `ComGAN` and `PertG` is available in [ComGAN & PertG](https://drive.google.com/file/d/1Hk-oraxtStH16BPf_2dveMdrncTSJOcI/view?usp=drive_link). Put them in `DF-RAP/checkpoints/`  .

The pretrained model of `SimSwap` and `Arcface` is available in [SimSwap](https://drive.google.com/drive/folders/1tGqLa87UogpMoDbzthsclIcL52-jHbk_?usp=drive_link). Put them in `DF-RAP/SimSwap/arcface_model/`  and  `DF-RAP/SimSwap/checkpoints/` .

The pretrained model of `StarGAN` is available in [StarGAN](https://www.dropbox.com/s/zdq6roqf63m0v5f/celeba-256x256-5attrs.zip?dl=0). Put it in `DF-RAP/checkpoints/stargan_celeba_256/models/`.

### Acknowledges

Our work is based on:

[1] https://github.com/mlomnitz/DiffJPEG

[2] https://github.com/yunjey/stargan

[3] https://github.com/neuralchen/SimSwap

### Dataset
https://github.com/ZOMIN28/OSN-transmission_mini_CelebA

### Visualization
#### PGD-Based
<img src="images\output.png" alt="output" style="zoom:67%;" />

#### Generator-Based
<img src="images\output2.png" alt="output2" style="zoom:67%;" />

### Citation

```
@article{qu2024df,
  title={DF-RAP: A Robust Adversarial Perturbation for Defending against Deepfakes in Real-world Social Network Scenarios},
  author={Qu, Zuomin and Xi, Zuping and Lu, Wei and Luo, Xiangyang and Wang, Qian and Li, Bin},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024},
  publisher={IEEE}
}
```
