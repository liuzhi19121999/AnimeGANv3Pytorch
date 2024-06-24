from torch.utils.data.dataset import Dataset
from torch import Tensor
from random import randint
import numpy as np
import os
import cv2

class ImageDataSet(Dataset):
    def __init__(self, img_dir: str, img_size=(256, 256), resize="resize") -> None:
        self.img_dir = img_dir
        # self.img_size = img_size
        # self.resize = resize

        self.photo = 'train_photo'
        self.style = 'style'
        self.smooth = 'smooth'
        self.smooth_noise = 'smooth_noise'

        self.photo_data = []
        self.style_data = []
        self.smooth_data = []
        self.smooth_noise_data = []

        self.photo_data_lenght = 0
        self.style_data_length = 0
        self.smooth_data_length = 0
        self.smooth_noise_data_length = 0

        self.init_data()
    
    def init_data(self):
        # dataset_dir = "./dataset"
        dataset_dir = "/kaggle/input/animegan/dataset"
        # dataset_dir = "/kaggle/input/animeganthe/dataset"
        # train photo
        train_dir = f"{dataset_dir}/train_photo"
        train_path_files = os.listdir(train_dir)
        for train_file in train_path_files:
            self.photo_data.append(f"{train_dir}/{train_file}")
        # style data
        style_dir = f"{dataset_dir}/{self.img_dir}/style"
        style_path_files = os.listdir(style_dir)
        for style_file in style_path_files:
            self.style_data.append(f"{style_dir}/{style_file}")
        # smooth data
        smooth_dir = f"{dataset_dir}/{self.img_dir}/smooth"
        smooth_path_files = os.listdir(smooth_dir)
        for smooth_file in smooth_path_files:
            self.smooth_data.append(f"{smooth_dir}/{smooth_file}")
        # smooth noise data
        smooth_noise_dir = f"{dataset_dir}/{self.img_dir}/smooth_noise"
        smooth_noise_path_files = os.listdir(smooth_noise_dir)
        self.smooth_noise_data = [f"{smooth_noise_dir}/{smooth_noise_file}" 
                                  for smooth_noise_file in smooth_noise_path_files]
        
        # lenght
        self.photo_data_lenght = self.photo_data.__len__() - 1
        self.style_data_length = self.style_data.__len__() - 1
        self.smooth_data_length = self.smooth_data.__len__() - 1
        self.smooth_noise_data_length = self.smooth_noise_data.__len__() - 1
    
    def load_photo(self, path: str) -> np.ndarray:
        '''载入图像 [c, h, w]'''
        if 'style' in path or 'smooth' in path:
            # color image1
            image = cv2.imread(path)
            image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

            image2 = np.zeros(image1.shape).astype(np.float32)
        else:
            # real photo
            image = cv2.imread(path)
            image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            # Color segmentation (ie. region smooth) photo
            image = cv2.imread(path.replace('train_photo', "seg_train_5-0.8-50"))
            image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image1 = image1.transpose((2, 0, 1))
        image2 = image2.transpose((2, 0, 1))

        image1 = image1 / 127.5 - 1.0
        image2 = image2 / 127.5 - 1.0
        return image1, image2
    
    def __len__(self):
        return self.photo_data_lenght or self.style_data_length or \
            self.smooth_data_length or self.smooth_noise_data_length
    
    def __getitem__(self, index) -> dict[str, Tensor]:
        photo_index = randint(0, self.photo_data_lenght)
        style_index = randint(0, self.style_data_length)
        smooth_index = randint(0, self.smooth_data_length)
        smooth_noise_index = randint(0, self.smooth_noise_data_length)

        # print(photo_index, style_index, smooth_index, smooth_noise_index)

        photo, _ = self.load_photo(self.photo_data[photo_index])
        style, _ = self.load_photo(self.style_data[style_index])
        smooth, _ = self.load_photo(self.smooth_data[smooth_index])
        smooth_noise, _ = self.load_photo(self.smooth_noise_data[smooth_noise_index])

        return {
            "photo": photo,
            "style": style,
            "smooth": smooth,
            "smooth_noise": smooth_noise
        }