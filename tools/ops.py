import torch
from torch import Tensor
import torch.nn.functional as F
from tools.vgg19 import VGG19Multi, VGG19Single
from tools.color_ops import *

vgg19_single = VGG19Single().eval()
vgg19_multi = VGG19Multi().eval()

def vgg_to_device(device: str):
    vgg19_single.to(device=device)
    vgg19_multi.to(device=device)

def con_loss_fn(real, fake, weight=1.0):
    return  weight * VGG_LOSS(real, fake)

def VGG_LOSS(x: Tensor, y: Tensor):
    # The number of feature channels in layer 4-4 of vgg19 is 512
    x: Tensor = vgg19_single(x)
    y: Tensor = vgg19_single(y)
    c = 0
    c = x.shape[0]
    if x.shape.__len__() == 4:
        c = x.shape[1]
    return  L1_loss(x, y)/torch.tensor(c, dtype=torch.float32)

def L1_loss(x, y):
    loss = torch.mean(torch.abs(x - y))
    return loss


def style_loss_decentralization_3(style, fake, weight):
    # [b, c, w, h]
    style_4, style_3, style_2 = vgg19_multi(style)
    fake_4, fake_3, fake_2 = vgg19_multi(fake)
    dim = [2, 3]
    style_2 = style_2 - torch.mean(style_2, dim=dim, keepdim=True)
    fake_2 = fake_2 - torch.mean(fake_2, dim=dim, keepdim=True)
    c_2 = fake_2.shape[1]

    style_3 = style_3 - torch.mean(style_3, dim=dim, keepdim=True)
    fake_3 = fake_3 - torch.mean(fake_3, dim=dim, keepdim=True)
    c_3 = fake_3.shape[1]

    style_4 = style_4 - torch.mean(style_4, dim=dim, keepdim=True)
    fake_4 = fake_4 - torch.mean(fake_4, dim=dim, keepdim=True)
    c_4 = fake_4.shape[1]

    loss4_4 = L1_loss(gram(style_4), gram(fake_4))/torch.tensor(c_4, dtype=torch.float32)
    loss3_3 = L1_loss(gram(style_3), gram(fake_3))/torch.tensor(c_3,dtype=torch.float32)
    loss2_2 = L1_loss(gram(style_2), gram(fake_2))/torch.tensor(c_2, dtype=torch.float32)
    return  weight[0] * loss2_2, weight[1] * loss3_3, weight[2] * loss4_4

def gram(x: Tensor):
    x = x.permute((0, 2, 3, 1))
    shape_x = x.shape
    b = shape_x[0]
    c = shape_x[3]
    x = x.reshape((b, -1, c))
    return torch.matmul(x.permute((0, 2, 1)), x) / torch.tensor((x.numel() // b), dtype=torch.float32)

def region_smoothing_loss(seg, fake, weight):
    return VGG_LOSS(seg, fake) * weight


def Lab_color_loss(photo, fake, weight=1.0):
    photo = (photo + 1.0) / 2.0
    fake = (fake + 1.0) / 2.0
    photo = rgb_to_lab(photo)
    fake = rgb_to_lab(fake)
    # L: 0~100, a: -128~127, b: -128~127
    loss = 2.0 * L1_loss(photo[:,:,:,0]/100.0, fake[:,:,:,0]/100.0) + L1_loss((photo[:,:,:,1]+128.0)/255.0, (fake[:,:,:,1]+128.0)/255.0) \
            + L1_loss((photo[:,:,:,2]+128.0)/255.0, (fake[:,:,:,2]+128.0)/255.0)
    return  weight * loss


def total_variation_loss(inputs):
    """
    [n, c, h, w]
    A smooth loss in fact. Like the smooth prior in MRF.
    V(y) = || y_{n+1} - y_n ||_2
    """
    # dh = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]
    # dw = inputs[:, :, :, :-1] - inputs[:, :, :, 1:]
    # size_dh = torch.tensor(torch.tensor(dh).size(), dtype=torch.float32)
    # size_dw = torch.tensor(torch.tensor(dw).size(), dtype=torch.float32)
    return F.mse_loss(inputs[:, :, :-1, :], inputs[:, :, 1:, :]) + F.mse_loss(inputs[:, :, :, :-1], inputs[:, :, :, 1:])


def generator_loss(fake):
    fake_loss = torch.mean(torch.square(fake - 0.9))
    return fake_loss


def discriminator_loss(anime_logit, fake_logit):
    # lsgan :
    anime_gray_logit_loss = torch.mean(torch.square(anime_logit - 0.9))
    fake_gray_logit_loss = torch.mean(torch.square(fake_logit - 0.1))
    # loss =   0.5 * anime_gray_logit_loss  \ # Hayao
    loss =   0.5 * anime_gray_logit_loss  \
           + 1.0 * fake_gray_logit_loss
    return loss


def discriminator_loss_346(fake_logit):
    # lsgan :
    fake_logit_loss = torch.mean(torch.square(fake_logit- 0.1))
    loss =  1.0 * fake_logit_loss
    return loss


def generator_loss_m(fake):
    loss = torch.mean(torch.square(fake - 1.0))
    return loss


def discriminator_loss_m(real, fake):
    real_loss = torch.mean(torch.square(real - 1.0))
    fake_loss = torch.mean(torch.square(fake))
    loss = real_loss + fake_loss
    return loss