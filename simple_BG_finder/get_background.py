import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#读取alpha
alpha_path = 'D:/A/Python/Datasets/CV/Segmentation/PersonSeg/png/'
alpha_dirs = os.listdir(alpha_path)

#读取image
img_path = 'D:/A/Python/Datasets/CV/Segmentation/PersonSeg/jpg/'
save_path = 'D:/A/Python/Datasets/CV/Segmentation/PersonSeg/background/'
img_dirs = os.listdir(img_path)

for alpha_dir,img_dir in zip(alpha_dirs[6524:6526],img_dirs[6524:6526]):
    alpha = plt.imread(alpha_path+alpha_dir)
    h,w = alpha.shape
    invert_array = np.ones((h,w))
    alpha_invert = invert_array-alpha
    
    img = plt.imread(img_path+img_dir)

    #相乘得到每个通道的背景
    c1 = alpha_invert*img[:,:,0]
    c2 = alpha_invert*img[:,:,1]
    c3 = alpha_invert*img[:,:,2]
    #叠加
    background = np.array([c1,c2,c3])
    background = background.swapaxes(0,2)
    background = background.swapaxes(0,1)
    #保存
    background = background.astype(np.uint8)
    plt.imsave(save_path+str(img_dir)+'.png',background)
    print(img_dir)