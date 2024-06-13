from DTGAN import GeneratorV3, DiscrimeV3
from tools.color_ops import rgb_to_lab
from tools.L0_smoothing import L0Smoothing
from tools.ops import con_loss
from tools.GuidedFilter import guided_filter
from torch import optim, Tensor
from torchvision.transforms.v2.functional import rgb_to_grayscale
import numpy as np
import torch
from skimage import segmentation, color
from joblib import Parallel, delayed
import cv2

def grayscale_to_rgb(input_tensor: Tensor) -> Tensor:
    if input_tensor.shape.__len__() == 4:
        return input_tensor.expand((-1, 3, -1, -1))
    return input_tensor.expand((3, -1, -1))

class Trainer:
    def __init__(self, dataset="Hayao", lr_g=2e-5, lr_d=4e-5, device="cpu"):
        super(Trainer, self).__init__()
        self.device_type = device
        self.G = GeneratorV3(dataset=dataset)
        self.D = DiscrimeV3()
        self.optimizer_g = optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    def train(self):
        '''训练模型'''
        photo = torch.rand((2, 3, 512, 512))
        real_photo = torch.rand((2, 3, 512, 512))

        anime = torch.rand((2, 3, 512, 512))
        anime_smooth = torch.rand((2, 3, 512, 512))

        fake_NLMean_l0 = torch.rand((2, 3, 512, 512))
        # 一次迭代
        # generated and discrimination
        generated_s, generated_m = self.G(photo)
        generated = self.tanh_out_scale(guided_filter(self.sigm_out_scale(generated_s),self.sigm_out_scale(generated_s), 2, 0.01))
        # val
        val_generated_s, val_generated_m = self.G(real_photo)
        real_generated = self.tanh_out_scale(guided_filter(self.sigm_out_scale(val_generated_s),self.sigm_out_scale(val_generated_s), 2, 0.01))
        # gray maping
        fake_sty_gray = grayscale_to_rgb(rgb_to_grayscale(generated))
        anime_sty_gray = grayscale_to_rgb(rgb_to_grayscale(anime))
        gray_anime_smooth = grayscale_to_rgb(rgb_to_grayscale(anime_smooth))
        # support
        fake_gray_logit = self.D(fake_sty_gray)
        anime_gray_logit = self.D(anime_sty_gray)
        gray_anime_smooth_logit = self.D(gray_anime_smooth)
        # main
        generated_m_logit = self.D(generated_m)
        fake_NLMean_logit = self.D(fake_NLMean_l0)
        # loss
        Pre_train_G_loss = con_loss(real_photo, generated)
        # GAN Support
        con_loss =  con_loss(real_photo, generated, 0.5)
        s22, s33, s44  = style_loss_decentralization_3(self.anime_sty_gray, self.fake_sty_gray,  [0.1, 2.0,  28]) 
        self.sty_loss = self.s22 + self.s33 + self.s44

        self.rs_loss =  region_smoothing_loss(self.fake_superpixel, self.generated, 0.8 ) + \
                        VGG_LOSS(self.photo_superpixel, self.generated) * 0.5

        self.color_loss =  Lab_color_loss(self.real_photo, self.generated, 8. )
        self.tv_loss  = 0.0001 * total_variation_loss(self.generated)

        self.g_adv_loss = generator_loss(fake_gray_logit)
        self.G_support_loss = self.g_adv_loss + self.con_loss + self.sty_loss   + self.rs_loss +  self.color_loss +self.tv_loss
        self.D_support_loss = discriminator_loss(anime_gray_logit, fake_gray_logit) \
                            + discriminator_loss_346(gray_anime_smooth_logit) * 5.
    
    def get_seg(self, batch_image):
        def get_superpixel(image):
            image = (image + 1.) * 127.5
            image = np.clip(image, 0, 255).astype(np.uint8)  # [-1. ,1.] ~ [0, 255]
            image_seg = segmentation.felzenszwalb(image, scale=5, sigma=0.8, min_size=100)
            image = color.label2rgb(image_seg, image,  bg_label=-1, kind='avg').astype(np.float32)
            image = image / 127.5 - 1.0
            return image
        num_job = np.shape(batch_image)[0]
        batch_out = Parallel(n_jobs=num_job)(delayed(get_superpixel) (image) for image in batch_image)
        return np.array(batch_out)

    def get_simple_superpixel_improve(self, batch_image, seg_num=200):
        def process_slic(image):
            seg_label = segmentation.slic(image, n_segments=seg_num, sigma=1, start_label=0,compactness=10, convert2lab=True)
            image = color.label2rgb(seg_label, image, bg_label=-1, kind='avg')
            return image
        num_job = np.shape(batch_image)[0]
        batch_out = Parallel(n_jobs=num_job)(delayed(process_slic)(image )for image in batch_image)
        return np.array(batch_out)

    def get_NLMean_l0(self, batch_image, ):
        def process_slic(image):
            image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
            image = cv2.fastNlMeansDenoisingColored(image, None, 7, 6, 6, 7)
            image = L0Smoothing(image/255, 0.005).astype(np.float32) * 2. - 1.
            return image.clip(-1., 1.)
        num_job = np.shape(batch_image)[0]
        batch_out = Parallel(n_jobs=num_job)(delayed(process_slic)(image) for image in batch_image)
        return np.array(batch_out)
    
    def to_lab(self, x):
        """
        @param x: image tensor  [-1.0, 1.0]
        # @return:  image tensor  [-1.0, 1.0]
        @return:  image tensor  [0.0, 1.0]
        """
        x = (x + 1.0) / 2.0
        x = rgb_to_lab(x)
        y = torch.concat([torch.unsqueeze(x[:, :, :, 0] / 100.,-1), torch.unsqueeze((x[:, :, :, 1]+128.)/255.,-1), torch.unsqueeze((x[:, :, :, 2]+128.)/255.,-1)], dim=-1)
        return y

    def sigm_out_scale(self, x):
        """
        @param x: image tensor  [-1.0, 1.0]
        @return:  image tensor  [0.0, 1.0]
        """
        # [-1.0, 1.0]  to  [0.0, 1.0]
        x = (x + 1.0) / 2.0
        return  torch.clamp(x, 0.0, 1.0)

    def tanh_out_scale(self, x):
        """
        @param x: image tensor  [0.0, 1.0]
        @return:  image tensor  [-1.0, 1.0]
        """
        # [0.0, 1.0]   to  [-1.0, 1.0]
        x = (x - 0.5) * 2.0
        return  torch.clamp(x,-1.0, 1.0)