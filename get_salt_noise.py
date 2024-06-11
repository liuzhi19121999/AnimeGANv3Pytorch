import numpy as np
import os, cv2,time
from tqdm import tqdm

opj = os.path.join

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def sp_noise(img, prob = 0.4):
    '''
    Add salt and pepper noise
     prob: noise ratio
    '''
    output = img.copy()
    h, w = output.shape[:2]
    sp = h*w
    NP = int(sp*(prob))
    for i in range(NP):
        randy = np.random.randint(1, h-1)
        randx = np.random.randint(1, w-1)
        output[randy, randx] = 255
    return output


if __name__ == "__main__":
    dataSet = "SummerWar"

    image_foder = f'./dataset/{dataSet}/smooth'
    out_foder = f'./dataset/{dataSet}/smooth_noise'
    check_folder(out_foder)
    imgs = os.listdir(image_foder)
    for x in tqdm(imgs):
        img =  cv2.imread(opj(image_foder,x))
        S = sp_noise(img, 0.1)
        cv2.imwrite(opj(out_foder,x),S)
