import torch.nn as nn
import torch
from torch import Tensor
from torchvision.models import vgg19
# import numpy as np

# data_dict: dict = np.load("./vgg19_no_fc.npy", encoding="latin1", allow_pickle=True).item()
# data_keys = []
# for i in data_dict:
#     data_keys.append(i)
# data_keys.sort()
# print(data_dict)

# data_dict_new = {}
# index = -1
# for layer in ["0", "2", "5", "7", "10", "12", "14", "16",
#               "19", "21", "23", "25", "28", "30", "32", "34"]:
#     index += 1
#     temp = data_dict[data_keys[index]]
#     data_dict_new[f"{layer}.weight"] = torch.tensor(temp[0]).permute((3, 2, 1, 0))
#     data_dict_new[f"{layer}.bias"] = torch.tensor(temp[1])

# vggmodels = vgg19(pretrained=False).features
# vggmodels.load_state_dict(data_dict_new)

# torch.save(vggmodels.to(device="cpu").state_dict(), "vgg19.pth")

vgg19_data = torch.load("./vgg19.pth")
vgg19_fea = vgg19(pretrained=False).features.eval()
vgg19_fea.load_state_dict(vgg19_data)
vggmodel = list(vgg19_fea.children())

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
            *vggmodel[:9]
        )
        self.first_conv = vggmodel[7]
        
        self.secnod_blk = nn.Sequential(
            *vggmodel[9:14]
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
    # print(vgg19_fea)
    vgg_model = VGG19Multi()
    # print(vgg_model)
    # print(vgg_model)
    imgs = torch.rand((2, 3, 224, 224))
    print(imgs.shape)
    outs0, out1, out2 = vgg_model(imgs)
    # print(outs)
    print(outs0.shape, " ", out1.shape, " ", out2.shape)
    # print("测试")