from DTGAN import GeneratorV3
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import cv2
import numpy as np

generate = GeneratorV3()
g_data = torch.load("./model_state/ChinaPhoto/generator.pth")
generate.load_state_dict(g_data["model"])

input_x = torch.randn((1, 3, 512, 512))

class OutModel(nn.Module):
    def __init__(self):
        super(OutModel, self).__init__()
        self.header_first = generate.header_first
        self.header_second = generate.header_second
        self.header_third = generate.header_third
        self.header_forth = generate.header_forth

        self.m_extral_attention = generate.m_extral_attention
        self.m_rb_first_f = generate.m_rb_first_f
        self.m_rb_first_s = generate.m_rb_first_s
        self.m_rb_second_f = generate.m_rb_second_f
        self.m_rb_second_s = generate.m_rb_second_s
        self.m_rb_third_f = generate.m_rb_third_f
        self.m_rb_third_s = generate.m_rb_third_s
        self.m_conv_tahn = generate.m_conv_tahn
    
    def padding_input(self, x: Tensor, k=1, s=1, padding_mode="reflect"):
        if (k - s) % 2 == 0 :
            pad = (k - s) // 2
            pad_top, pad_bottom, pad_left, pad_right = pad, pad, pad, pad

        else :
            pad = (k - s) // 2
            pad_bottom, pad_right = pad, pad,
            pad_top, pad_left = k - s - pad_bottom, k - s - pad_right

        if padding_mode == 'zero' :
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        if padding_mode == 'reflect' :
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        return x

    def forward(self, x: Tensor):
        x0 = self.header_first(x)
        x1 = self.header_second(x0)
        x2 = self.header_third(x1)
        x3 = self.header_forth(x2)

        m_x3 = self.m_extral_attention(x3)
        m_x4 = self.m_rb_first_f(m_x3)
        m_x4 = self.m_rb_first_s(m_x4 + x2)

        m_x5 = self.m_rb_second_f(m_x4)
        m_x5 = self.m_rb_second_s(m_x5 + x1)

        m_x6 = self.m_rb_third_f(m_x5)
        m_x6 = self.m_rb_third_s(m_x6 + x0)

        m_x6 = self.padding_input(m_x6, 7)
        fake_m = self.m_conv_tahn(m_x6)
        return fake_m

out_model = OutModel()

def load_photo(path: str) -> np.ndarray:
        '''载入图像 [c, h, w]'''
        # color image1
        image = cv2.imread(path)
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image1 = image1.transpose((2, 0, 1))
        image1 = image1 / 127.5 - 1.0
        return image1

def show_img(img_data: torch.Tensor):
        print(img_data)
        print(img_data.shape)
        img_data = (img_data + 1.0) * 127.5
        img_data.clamp(0, 255)
        img_data = img_data[0].permute((1, 2, 0)).detach().numpy().astype(np.uint8)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR).astype(np.uint8)
        cv2.imshow("img", img_data)
        cv2.waitKey(0)

torch.onnx.export(
    out_model,
    input_x,
    "./generate.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"}
    }
)