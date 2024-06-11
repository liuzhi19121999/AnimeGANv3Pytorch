import torch.nn as nn
import torch
from torch import Tensor
from torchvision.models import vgg19

vggmodel = list(vgg19(pretrained=True).features.children())

VGG_MEAN = [103.939, 116.779, 123.68]

class VGG19Single(nn.Module):
    '''VGG19 单输出模型'''
    def __init__(self):
        super(VGG19Single, self).__init__()
        self.model = nn.Sequential(
            *vggmodel[:26]
        )
    
    def forward(self, x):
        rgb_scaled = ((x + 1) / 2) * 255.0
        red, green, blue = torch.split(rgb_scaled, 1, 1)
        bgr = torch.concat(dim=1, tensors=[blue - VGG_MEAN[0],
                                        green - VGG_MEAN[1],
                                        red - VGG_MEAN[2]])
        return self.model(bgr)

class VGG19Multi(nn.Module):
    '''VGG19 多输出模型'''
    def __init__(self):
        super(VGG19Multi, self).__init__()
        
        self.first_blk = nn.Sequential(
            *vggmodel[:7]
        )
        self.first_conv = vggmodel[7]
        
        self.secnod_blk = nn.Sequential(
            *vggmodel[7:14]
        )
        self.second_conv = vggmodel[14]

        self.last_blk = nn.Sequential(
            *vggmodel[14:26]
        )
    
    def forward(self, x: Tensor):
        '''input format: bgr image with shape [batch_size, 3, h, w] !!!
        scale: (-1, 1)'''
        rgb_scaled = ((x + 1) / 2) * 255.0
        red, green, blue = torch.split(rgb_scaled, 1, 1)
        bgr = torch.concat(dim=1, tensors=[blue - VGG_MEAN[0],
                                        green - VGG_MEAN[1],
                                        red - VGG_MEAN[2]])
        conv2_2 = self.first_blk(bgr)
        conv2_2_no_relu = self.first_conv(conv2_2)
        conv3_2 = self.secnod_blk(conv2_2)
        conv3_3_no_relu = self.second_conv(conv3_2)
        conv4_4_no_relu = self.last_blk(conv3_2)
        return conv4_4_no_relu, conv3_3_no_relu, conv2_2_no_relu

if __name__ == "__main__":
    vgg_model = VGG19Multi()
    print(vgg_model)
    imgs = torch.rand((2, 3, 224, 224))
    print(imgs.shape)
    outs0, out1, out2 = vgg_model(imgs)
    # print(outs)
    print(outs0.shape, " ", out1.shape, " ", out2.shape)
    print("测试")