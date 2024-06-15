import os
import numpy as np
import cv2, os, argparse
from glob import glob
from tqdm import tqdm

def parse_args():
    desc = "Edge smoothed"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='Hayao', help='dataset_name')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')

    return parser.parse_args()

def guass_init(kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    return kernel_size, kernel, gauss

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def make_edge_smooth(bgr_img, gray_img, img_size, kernel_size, kernel, gauss) :

    bgr_img = cv2.resize(bgr_img, (img_size, img_size))
    pad_img = np.pad(bgr_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
    gray_img = cv2.resize(gray_img, (img_size, img_size))

    edges = cv2.Canny(gray_img, 100, 200)
    dilation = cv2.dilate(edges, kernel)

    gauss_img = np.copy(bgr_img)
    idx = np.where(dilation != 0)
    for i in range(np.sum(dilation != 0)):
        gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
            np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
        gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
            np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
        gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
            np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

    return gauss_img

"""main"""
def main():
    # parse arguments
    args = parse_args()

    kernel_size, kernel, gauss = guass_init()

    dataset_name = args.dataset
    check_folder(os.path.dirname(os.path.dirname(__file__)) + '/dataset/{}/{}'.format(dataset_name, 'smooth'))
    file_list = glob(os.path.dirname(os.path.dirname(__file__)) + '/dataset/{}/{}/*.*'.format(dataset_name, 'style'))
    save_dir = os.path.dirname(os.path.dirname(__file__)) + '/dataset/{}/smooth'.format(dataset_name)

    for f in tqdm(file_list):
        file_name = os.path.basename(f)

        bgr_img = cv2.imread(f)
        gray_img = cv2.imread(f, 0)
        gauss_img = make_edge_smooth(bgr_img, gray_img, args.img_size, kernel_size, kernel, gauss)
        cv2.imwrite(os.path.join(save_dir, file_name), gauss_img)


if __name__ == '__main__':
    main()
