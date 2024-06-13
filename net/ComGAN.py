import torch
import torch.nn as nn
from DiffJPEG.DiffJPEG import DiffJPEG
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
        self.conv1x1 = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False)
    def forward(self, x):
        return self.conv1x1(x) + self.main(x)

# Down sampling module
def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU(),
        nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU(),
    )

# Up sampling module
def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )

class ComGenerator(nn.Module):
    def __init__(self,dim_in=3, dim_out=32, isJPEG=False):
        super(ComGenerator, self).__init__()
        self.isJPEG = isJPEG
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
    
            )
        self.conv2 = nn.Sequential(
            ResidualBlock(dim_out,dim_out*2),
    
            )
        self.conv3 = nn.Sequential(
            ResidualBlock(dim_out*2,dim_out*4),
    
            )
        self.conv4 = nn.Sequential(
            ResidualBlock(dim_out*4,dim_out*8),

            )
        self.conv5 = nn.Sequential(
            ResidualBlock(dim_out*8,dim_out*16),
            )

        self.conv4m = nn.Sequential(
            ResidualBlock(dim_out*16, dim_out*8),
            )
        self.conv3m = nn.Sequential(
            ResidualBlock(dim_out*8, dim_out*4),
            )
        self.conv2m = nn.Sequential(
            ResidualBlock(dim_out*4, dim_out*2),
            )
        self.conv1m = nn.Sequential(
            ResidualBlock(dim_out*2, dim_out),
            )
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(dim_out, 3, 3, 1, 1),
            nn.Tanh()
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.upsample54 = upsample(dim_out*16, dim_out*8)
        self.upsample43 = upsample(dim_out*8, dim_out*4)
        self.upsample32 = upsample(dim_out*4, dim_out*2)
        self.upsample21 = upsample(dim_out*2, dim_out)

        self.qf,self.freq = self.read_qf()
        self.q = self.qf_sample()

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv4_out = self.conv4(self.max_pool(conv3_out))
        conv5_out = self.conv5(self.max_pool(conv4_out))

        conv5m_out = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
        conv4m_out = self.conv4m(conv5m_out)
        conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
        conv3m_out = self.conv3m(conv4m_out_)
        conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
        conv2m_out = self.conv2m(conv3m_out_)
        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
        conv1m_out = self.conv1m(conv2m_out_)
        conv0_out = self.conv0(conv1m_out)

        if self.isJPEG:
            q = self.qf_sample()
            if q != 100:
                conv0_out = DiffJPEG(height=256, width=256, quality=q, differentiable=True).to(device)((conv0_out + 1) / 2)

        return conv0_out


    def qf_sample(self):
        return int(self.qf[torch.multinomial(self.freq, 1).item()].item())
    
    
    def read_qf(self):
        with open('data/qf.txt', 'r') as file:
            lines = file.readlines()
            qf = torch.tensor([float(x) for x in lines[0].strip().split(',')])
            freq = torch.tensor([float(x) for x in lines[1].strip().split(',')])
            return qf,freq
        