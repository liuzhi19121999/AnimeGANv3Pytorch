import argparse, os, cv2
from tqdm import tqdm
from edge_smooth import guass_init, check_folder, make_edge_smooth
from get_salt_noise import sp_noise, opj
from visual_super_seg import get_superPixel
import numpy as np

def parse_args():
    desc = "Preparing For Training"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='shinkai', help='dataset_name')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')

    return parser.parse_args()

def edge_smooth_func(args):
    # parse arguments
    # args = parse_args()
    kernel_size, kernel, gauss = guass_init()
    dataset_name = args.dataset
    # print(os.path.dirname(os.path.dirname(__file__)) + '/dataset/{}/{}'.format(dataset_name, 'smooth'))
    check_folder('./dataset/{}/{}'.format(dataset_name, 'smooth'))
    file_list = ['./dataset/{}/{}/{}'.format(dataset_name, 'style', file) for file in os.listdir('./dataset/{}/{}'.format(dataset_name, 'style'))]
    save_dir = './dataset/{}/smooth'.format(dataset_name)

    for f in tqdm(file_list):
        file_name = os.path.basename(f)
        bgr_img = cv2.imread(f)
        gray_img = cv2.imread(f, 0)
        gauss_img = make_edge_smooth(bgr_img, gray_img, args.img_size, kernel_size, kernel, gauss)
        cv2.imwrite(os.path.join(save_dir, file_name), gauss_img)

def gen_salt_noise(dataSet: str):
    # dataSet = "SummerWar"
    image_foder = f'./dataset/{dataSet}/smooth'
    out_foder = f'./dataset/{dataSet}/smooth_noise'
    check_folder(out_foder)
    imgs = os.listdir(image_foder)
    for x in tqdm(imgs):
        img =  cv2.imread(opj(image_foder,x))
        S = sp_noise(img, 0.1)
        cv2.imwrite(opj(out_foder,x),S)

def visual_super_seg():
    temp = './dataset/seg_train_5-0.8-50'
    # temp = '../dataset/seg_slic_train_1000'
    check_folder(temp)
    # image_foder = '../dataset/val'
    # image_foder = '../dataset/Hayao/style'
    image_foder = './dataset/train_photo'

    for i, x in enumerate(os.listdir(image_foder)):
        print(i, x)
        # if x != '2013-11-10 12_45_41.jpg':
        # if x != '4.jpg':
        #     continue
        path = os.path.join(image_foder,x)
        img = get_superPixel(path)
        # img = get_simple_superpixel_improve(path, 1000)
        cv2.imwrite(os.path.join(temp,x), cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2BGR))
        # cv2.imshow('super_seg',cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)

def main():
    args = parse_args()
    edge_smooth_func(args=args)
    gen_salt_noise(args.dataset)
    visual_super_seg()

if __name__ == "__main__":
    main()