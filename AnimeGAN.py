from DTGAN import GeneratorV3, DiscrimeV3
from tools.color_ops import rgb_to_lab
from tools.L0_smoothing import L0Smoothing
from tools.ops import con_loss_fn, style_loss_decentralization_3, region_smoothing_loss, \
    VGG_LOSS, Lab_color_loss, total_variation_loss, generator_loss, discriminator_loss, \
    discriminator_loss_346, L1_loss, generator_loss_m, discriminator_loss_m, vgg_to_device
from tools.GuidedFilter import guided_filter
from torch import optim, Tensor, nn
from torchvision.transforms.v2.functional import rgb_to_grayscale
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
from skimage import segmentation, color
from joblib import Parallel, delayed
from dataSet import ImageDataSet
from tools.utils import check_folder
import cv2
from time import time

def grayscale_to_rgb(input_tensor: Tensor) -> Tensor:
    if input_tensor.shape.__len__() == 4:
        return input_tensor.expand((-1, 3, -1, -1))
    return input_tensor.expand((3, -1, -1))

class Trainer:
    def __init__(self, dataset="SummerWar", epoch=100, start_epoch=0, batch=4, init_g_epoch=10, init_lr_g=2e-4, lr_g=1e-4, lr_d=1e-4, device="cpu"):
        super(Trainer, self).__init__()
        self.epoch = epoch
        # self.data_set_num = data_set_num
        self.batch = batch
        self.init_g_epoch = init_g_epoch
        self.device_type = device
        self.data_dir = dataset
        self.start_epoch = start_epoch

        self.G = GeneratorV3(dataset=dataset)
        self.D = DiscrimeV3()
        self.optimizer_init_g = optim.Adam(self.G.parameters(), lr=init_lr_g, betas=(0.5, 0.999))
        self.optimizer_g = optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.999))
        self.data_set = ImageDataSet(dataset)

        self.load_model_data()
        self.model_to_device()
        vgg_to_device(device)
    
    def save_model_data(self):
        '''保存模型数据'''
        check_folder(f"./model_state/{self.data_dir}")
        # G
        torch.save({"model": self.G.to(device="cpu").state_dict()}, f"./model_state/{self.data_dir}/generator.pth")
        # D
        torch.save({"model": self.D.to(device="cpu").state_dict()}, f"./model_state/{self.data_dir}/discrime.pth")
        # 模型迁移回原设备
        self.model_to_device()
    
    def init_model_weight(self, m):
        # recommend
        if isinstance(m, nn.Conv2d): 
            nn.init.xavier_normal_(m.weight.data) 
            nn.init.xavier_normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias, 0)
    
    def load_model_data(self):
        # G
        try:
            g_state = torch.load(f"./model_state/{self.data_dir}/generator.pth")
            self.G.load_state_dict(g_state["model"])
        except Exception:
            self.init_model_weight(self.G)
        # D
        try:
            d_state = torch.load(f"./model_state/{self.data_dir}/discrime.pth")
            self.D.load_state_dict(d_state["model"])
        except Exception:
            self.init_model_weight(self.D)
    
    def model_to_device(self):
        self.D.to(device=self.device_type)
        self.G.to(device=self.device_type)
    
    def init_g_train(self, photo: Tensor):
        '''初始化生成器'''
        self.optimizer_init_g.zero_grad()
        generated_s, generated_m = self.G(photo.to(device=self.device_type))
        generated = self.tanh_out_scale(guided_filter(self.sigm_out_scale(generated_s),self.sigm_out_scale(generated_s), 2, 0.01))
        Pre_train_G_loss: Tensor = con_loss_fn(photo, generated)
        Pre_train_G_loss.backward()
        self.optimizer_init_g.step()
        return Pre_train_G_loss.detach()
    
    def update_g(self, photo: Tensor, photo_superpixel: Tensor, anime: Tensor, anime_smooth: Tensor):
        '''训练模型'''
        self.optimizer_d.zero_grad()
        self.optimizer_g.zero_grad()
        # Update G Net
        generated_s, generated_m = self.G(photo.to(device=self.device_type))
        generated: Tensor = self.tanh_out_scale(guided_filter(self.sigm_out_scale(generated_s),self.sigm_out_scale(generated_s), 2, 0.01))
        fake_superpixel = self.get_seg(generated.detach().to(device="cpu").numpy()).to(device=self.device_type)
        fake_NLMean_l0 = self.get_NLMean_l0(generated_s.detach().to(device="cpu").numpy()).to(device=self.device_type)
        fake_sty_gray = grayscale_to_rgb(rgb_to_grayscale(generated))
        anime_sty_gray = grayscale_to_rgb(rgb_to_grayscale(anime.to(device=self.device_type)))
        gray_anime_smooth = grayscale_to_rgb(rgb_to_grayscale(anime_smooth.to(device=self.device_type)))
        # support
        fake_gray_logit = self.D(fake_sty_gray.detach())
        anime_gray_logit = self.D(anime_sty_gray.detach())
        gray_anime_smooth_logit = self.D(gray_anime_smooth.detach())
        generated_m_logit = self.D(generated_m.detach())
        # GAN Support
        con_loss =  con_loss_fn(photo, generated, 0.5)
        s22, s33, s44  = style_loss_decentralization_3(anime_sty_gray, fake_sty_gray,  [0.1, 2.0,  28])
        sty_loss = s22 + s33 + s44
        rs_loss =  region_smoothing_loss(fake_superpixel, generated, 0.8 ) + \
                        VGG_LOSS(photo_superpixel, generated) * 0.5
        color_loss =  Lab_color_loss(photo, generated, 8. )
        tv_loss  = 0.0001 * total_variation_loss(generated)
        g_adv_loss = generator_loss(fake_gray_logit.detach())
        G_support_loss = g_adv_loss + con_loss + sty_loss + rs_loss + color_loss + tv_loss
        # main
        tv_loss_m = 0.0001 * total_variation_loss(generated_m)
        p4_loss = VGG_LOSS(fake_NLMean_l0, generated_m) * 0.5
        p0_loss = L1_loss(fake_NLMean_l0, generated_m) * 50
        g_m_loss = generator_loss_m(generated_m_logit.detach()) * 0.02

        G_main_loss = g_m_loss + p0_loss + p4_loss + tv_loss_m
        Generator_loss: Tensor =  G_support_loss +  G_main_loss
        # self.optimizer_g.zero_grad()
        Generator_loss.backward()
        self.optimizer_g.step()
        
        # Update D Net
        # main
        fake_NLMean_logit = self.D(fake_NLMean_l0)
        D_support_loss = discriminator_loss(anime_gray_logit, fake_gray_logit) \
                            + discriminator_loss_346(gray_anime_smooth_logit) * 5.
        D_main_loss = discriminator_loss_m(fake_NLMean_logit, generated_m_logit) * 0.1
        Discriminator_loss: Tensor = D_support_loss + D_main_loss
        # self.optimizer_d.zero_grad()
        Discriminator_loss.backward()
        self.optimizer_d.step()

        return Generator_loss.detach(), Discriminator_loss.detach()
    
    def train(self):
        '''训练模型'''
        for epo in range(self.start_epoch, self.epoch + 1):
            # Steps of Every Epoch
            data_loader = DataLoader(self.data_set, batch_size=self.batch, shuffle=False)
            step_length = data_loader.__len__()
            for (step, data_batch) in enumerate(data_loader):
                start_time = time()
                real_photo = torch.concat(data_batch["photo"])
                anime = torch.concat(data_batch["style"])
                anime_smooth = torch.concat(data_batch["smooth"])
                
                if epo < self.init_g_epoch:
                    g_loss = self.init_g_train(real_photo)
                    g_loss_print = g_loss.to(device="cpu").data
                    step_time = time() - start_time
                    print(f"Epoch: {epo:4d}  Step: {step:5d}/{step_length}  Time: {int(step_time):4d} s  ETA: {int((step_length - step - 1)*step_time):6d} s  G-Loss: {g_loss_print:10f}")
                else:
                    photo_superpixel = self.get_seg(real_photo.detach().numpy())
                    g_loss, d_loss = self.update_g(real_photo, photo_superpixel, anime, anime_smooth)
                    g_loss_print = g_loss.to(device="cpu").data
                    d_loss_print = d_loss.to(device="cpu").data
                    step_time = time() - start_time
                    print(f"Epoch: {epo:4d}  Step: {step:5d}/{step_length}  Time: {int(step_time):4d} s  ETA: {int((step_length - step - 1)*step_time):6d} s  G-Loss: {g_loss_print:10f}  D-Loss: {d_loss_print:10f}")
                if step % 30 == 0:
                    self.save_model_data()
            self.save_model_data()

        # photo = torch.rand((2, 3, 512, 512))
        # real_photo = torch.rand((2, 3, 512, 512))
        # photo_superpixel = torch.rand((2, 3, 512, 512))

        # anime = torch.rand((2, 3, 512, 512))
        # anime_smooth = torch.rand((2, 3, 512, 512))

        # fake_superpixel = torch.rand((2, 3, 512, 512))
        # fake_NLMean_l0 = torch.rand((2, 3, 512, 512))
        # # 一次迭代
        # # generated and discrimination
        # generated_s, generated_m = self.G(photo)
        # generated = self.tanh_out_scale(guided_filter(self.sigm_out_scale(generated_s),self.sigm_out_scale(generated_s), 2, 0.01))
        # # val
        # val_generated_s, val_generated_m = self.G(real_photo)
        # val_generated = self.tanh_out_scale(guided_filter(self.sigm_out_scale(val_generated_s),self.sigm_out_scale(val_generated_s), 2, 0.01))
        # # gray maping
        # fake_sty_gray = grayscale_to_rgb(rgb_to_grayscale(generated))
        # anime_sty_gray = grayscale_to_rgb(rgb_to_grayscale(anime))
        # gray_anime_smooth = grayscale_to_rgb(rgb_to_grayscale(anime_smooth))
        # # support
        # fake_gray_logit = self.D(fake_sty_gray)
        # anime_gray_logit = self.D(anime_sty_gray)
        # gray_anime_smooth_logit = self.D(gray_anime_smooth)
        # # main
        # generated_m_logit = self.D(generated_m)
        # fake_NLMean_logit = self.D(fake_NLMean_l0)
        # # loss
        # Pre_train_G_loss: Tensor = con_loss_fn(real_photo, generated)
        # # GAN Support
        # con_loss =  con_loss_fn(real_photo, generated, 0.5)
        # s22, s33, s44  = style_loss_decentralization_3(anime_sty_gray, fake_sty_gray,  [0.1, 2.0,  28]) 
        # sty_loss = s22 + s33 + s44

        # rs_loss =  region_smoothing_loss(fake_superpixel, generated, 0.8 ) + \
        #                 VGG_LOSS(photo_superpixel, generated) * 0.5

        # color_loss =  Lab_color_loss(real_photo, generated, 8. )
        # tv_loss  = 0.0001 * total_variation_loss(generated)

        # g_adv_loss = generator_loss(fake_gray_logit)
        # G_support_loss = g_adv_loss + con_loss + sty_loss + rs_loss + color_loss + tv_loss
        # D_support_loss = discriminator_loss(anime_gray_logit, fake_gray_logit) \
        #                     + discriminator_loss_346(gray_anime_smooth_logit) * 5.
        # # main
        # tv_loss_m = 0.0001 * total_variation_loss(generated_m)
        # p4_loss = VGG_LOSS(fake_NLMean_l0, generated_m) * 0.5
        # p0_loss = L1_loss(fake_NLMean_l0, generated_m) * 50
        # g_m_loss = generator_loss_m(generated_m_logit) * 0.02

        # G_main_loss = g_m_loss + p0_loss + p4_loss + tv_loss_m
        # D_main_loss = discriminator_loss_m(fake_NLMean_logit, generated_m_logit) * 0.1

        # Generator_loss: Tensor =  G_support_loss +  G_main_loss
        # Discriminator_loss: Tensor = D_support_loss + D_main_loss
    
    def get_seg(self, batch_image):
        def get_superpixel(image):
            image = (image + 1.) * 127.5
            image = np.clip(image, 0, 255).astype(np.uint8)  # [-1. ,1.] ~ [0, 255]
            image_seg = segmentation.felzenszwalb(image, scale=5, sigma=0.8, min_size=100)
            image = color.label2rgb(image_seg, image, bg_label=-1, kind='avg').astype(np.float32)
            image = image / 127.5 - 1.0
            return image
        num_job = np.shape(batch_image)[0]
        batch_out = Parallel(n_jobs=num_job)(delayed(get_superpixel)(image.transpose((1, 2, 0))) for image in batch_image)
        return torch.tensor(np.array(batch_out)).permute((0, 3, 1, 2))

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
        batch_out = Parallel(n_jobs=num_job)(delayed(process_slic)(image.transpose((1, 2, 0))) for image in batch_image)
        return torch.tensor(np.array(batch_out)).permute((0, 3, 1, 2))
    
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

if __name__ == "__main__":
    train = Trainer()
    train.train()