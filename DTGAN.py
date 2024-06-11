# 类 DTGAN 模型网络
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
# DTGAN 层结构
class ConLadeLRelu(nn.Module):
    '''Conv Lade LeakyRelu层'''
    def __init__(self, num_input: int, num_output: int, kernel=3, stride=1, padding_mode="reflect"):
        super(ConLadeLRelu, self).__init__()
        self.kernel_size = kernel
        self.stris = stride
        self.out_num = num_output
        self.padding_mode = padding_mode
        self.convd = nn.Conv2d(num_input, num_output, kernel, stride, bias=False, padding=0)
        self.lade_convd = nn.Conv2d(num_output, num_output, 1, 1, bias=False, padding=0)
        self.lrelu = nn.LeakyReLU(0.2)
    
    def lade(self, x: Tensor) -> Tensor:
        eps = 1e-5
        tx = self.padding_input(x)
        tx = self.lade_convd(tx)
        t_mean = torch.mean(tx, dim=[1, 2], keepdim=True)  
        # 计算方差（注意：PyTorch 的 var 默认是无偏的，即除以 N-1，但我们可以设置 correction=0 来得到样本方差，即除以 N）  
        t_sigma = torch.var(tx, dim=[1, 2], keepdim=True, unbiased=False)
        
        in_mean = torch.mean(x, dim=[1, 2], keepdim=True)
        # 计算方差（注意：PyTorch 的 var 默认是无偏的，即除以 N-1，但我们可以设置 correction=0 来得到样本方差，即除以 N）  
        in_sigma = torch.var(x, dim=[1, 2], keepdim=True, unbiased=False)
        x_in = (x - in_mean) / (torch.sqrt(in_sigma + eps))
        x = x_in * (torch.sqrt(t_sigma + eps)) + t_mean
        return x
    
    def padding_input(self, x: Tensor, k=1, s=1):
        if (k - s) % 2 == 0 :
            pad = (k - s) // 2
            pad_top, pad_bottom, pad_left, pad_right = pad, pad, pad, pad

        else :
            pad = (k - s) // 2
            pad_bottom, pad_right = pad, pad,
            pad_top, pad_left = k - s - pad_bottom, k - s - pad_right

        if self.padding_mode == 'zero' :
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        if self.padding_mode == 'reflect' :
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        return x
    
    def forward(self, x: Tensor):
        x = self.padding_input(x, self.kernel_size, self.stris)
        x = self.convd(x)
        x = self.lade(x)
        return self.lrelu(x)

class ExtraAttention(nn.Module):
    '''外部注意力层'''
    def __init__(self, input_num: int, out_num: int, kernel: int, stris: int, padding_mode="reflect"):
        super(ExtraAttention, self).__init__()
        self.padding_mode = padding_mode
        self.conv2d_1 = nn.Conv2d(input_num, out_num, kernel, stris)
        self.conv1d_1 = nn.Conv1d(out_num, out_num, kernel, stris, padding=0)
        self.conv1d_2 = nn.Conv1d(out_num, out_num, kernel_size=kernel, stride=stris, padding=0)
        self.conv2d_2 = nn.Conv2d(out_num, out_num, kernel_size=kernel, stride=stris)  
        self.batch_norm = nn.BatchNorm2d(out_num, eps=0.001, momentum=0.999)
        self.leRelu = nn.LeakyReLU()
    
    def padding_input(self, x: Tensor, k=1, s=1):
        if (k - s) % 2 == 0 :
            pad = (k - s) // 2
            pad_top, pad_bottom, pad_left, pad_right = pad, pad, pad, pad

        else :
            pad = (k - s) // 2
            pad_bottom, pad_right = pad, pad,
            pad_top, pad_left = k - s - pad_bottom, k - s - pad_right

        if self.padding_mode == 'zero' :
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        if self.padding_mode == 'reflect' :
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        return x
    
    def forward(self, x: Tensor):
        idn = x.detach()
        b, c, h, w = x.shape
        x = self.padding_input(x)
        x = self.conv2d_1(x)
        x = x.view(b, c, -1)  # reshape to [b, c, h*w]
        attn = self.conv1d_1(x)  # 1D conv, remove extra dim
        attn = F.softmax(attn, dim=2)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))
        x = self.conv1d_2(attn).view(b, c, h, w)  # reshape back to [b, c, h, w]
        x = self.padding_input(x)
        x = self.conv2d_2(x)
        x = self.batch_norm(x)
        x = x + idn
        out = self.leRelu(x)
        return out

class RBConvLeadeRelu(nn.Module):
    '''Bilinear Covnd 层'''
    def __init__(self, in_num: int, out_num: int, kernel: int, strides: int, scale_facotr=2.0):
        super(RBConvLeadeRelu, self).__init__()
        self.convLeade1 = ConLadeLRelu(in_num, out_num, kernel, strides)
        self.scale_factor = scale_facotr
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        x = self.convLeade1(x)
        return x
# DTGAN END

class GeneratorV3(nn.Module):
    '''生成器 DTGAN'''
    def __init__(self, dataset="") -> None:
        super(GeneratorV3, self).__init__()
        self.name = f"{self.__class__.__name__}_{dataset}"

        # header
        self.header_first = ConLadeLRelu(3, 32, 7, 1)
        self.header_second = nn.Sequential(
            ConLadeLRelu(32, 32, 3, 2),
            ConLadeLRelu(32, 64, 3, 1)
        )
        self.header_third = nn.Sequential(
            ConLadeLRelu(64, 64, 3, 2),
            ConLadeLRelu(64, 128, 3, 1)
        )
        self.header_forth = nn.Sequential(
            ConLadeLRelu(128, 128, 3, 2),
            ConLadeLRelu(128, 128, 3, 1)
        )

        # main blk
        self.m_extral_attention = ExtraAttention(128, 128, 1, 1)
        self.m_rb_first_f = RBConvLeadeRelu(128, 128, 3, 1)
        self.m_rb_first_s = ConLadeLRelu(128, 128, 3, 1)
        self.m_rb_second_f = RBConvLeadeRelu(128, 64, 3, 1)
        self.m_rb_second_s = ConLadeLRelu(64, 64, 3, 1)
        self.m_rb_third_f = RBConvLeadeRelu(64, 32, 3, 1)
        self.m_rb_third_s = ConLadeLRelu(32, 32, 3, 1)
        self.m_conv_tahn = nn.Sequential(
            nn.Conv2d(32, 3, 7, 1),
            nn.Tanh()
        )

        # support blk
        self.s_extral_attention = ExtraAttention(128, 128, 1, 1)
        self.s_rb_first_f = RBConvLeadeRelu(128, 128, 3, 1)
        self.s_rb_first_s = ConLadeLRelu(128, 128, 3, 1)
        self.s_rb_second_f = RBConvLeadeRelu(128, 64, 3, 1)
        self.s_rb_second_s = ConLadeLRelu(64, 64, 3, 1)
        self.s_rb_third_f = RBConvLeadeRelu(64, 32, 3, 1)
        self.s_rb_third_s = ConLadeLRelu(32, 32, 3, 1)
        self.s_conv_tahn = nn.Sequential(
            nn.Conv2d(32, 3, 7, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x0 = self.header_first(x)
        x1 = self.header_second(x0)
        x2 = self.header_third(x1)
        x3 = self.header_forth(x2)
        # support
        s_x3 = self.s_extral_attention(x3)
        s_x4 = self.s_rb_first_f(s_x3)
        s_x4 = self.s_rb_first_s(s_x4 + x2)

        s_x5 = self.s_rb_second_f(s_x4)
        s_x5 = self.s_rb_second_s(s_x5 + x1)

        s_x6 = self.s_rb_third_f(s_x5)
        s_x6 = self.s_rb_third_s(s_x6 + x0)

        fake_s = self.s_conv_tahn(s_x6)

        # main
        m_x3 = self.m_extral_attention(x3)
        m_x4 = self.m_rb_first_f(m_x3)
        m_x4 = self.m_rb_first_s(m_x4 + x2)

        m_x5 = self.m_rb_second_f(m_x4)
        m_x5 = self.m_rb_second_s(m_x5 + x1)

        m_x6 = self.m_rb_third_f(m_x5)
        m_x6 = self.m_rb_third_s(m_x6 + x0)

        fake_m = self.m_conv_tahn(m_x6)

        return fake_s, fake_m

class DiscrimeV3(nn.Module):
    '''判别器 DTGAN'''
    def __init__(self):
        super(DiscrimeV3, self).__init__()
        self.first_input = nn.Sequential(
            nn.Conv2d(3, 32, 7, 1),
            nn.LeakyReLU()
        )
        self.main = nn.Sequential(
            ConLadeLRelu(32, 64, 3, 2),
            ConLadeLRelu(64, 64, 3, 1),
            ConLadeLRelu(64, 128, 3, 2),
            ConLadeLRelu(128, 128, 3, 1),
            ConLadeLRelu(128, 256, 3, 2),
            ConLadeLRelu(256, 256, 3, 1)
        )
        self.end = nn.Sequential(
            nn.Conv2d(256, 1, 1, 1)
        )
    
    def forward(self, x):
        x = self.first_input(x)
        x = self.main(x)
        x = self.end(x)
        return x

if __name__ == "__main__":
    input_val = torch.rand((2, 3, 1024, 512))
    model = GeneratorV3("hayao")
    # dis = DiscrimeV3().to(device=dml)
    y_s, y_m = model(input_val)
    print(y_s)