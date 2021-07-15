import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil

# input_dir_jpg = 'D:/A/Python/Datasets/CV/Segmentation/PersonSeg/jpg/'
# save_dir_jpg = 'D:/A/Python/Datasets/CV/Segmentation/PersonSeg/train_white_jpg/'
# input_dir_png = 'D:/A/Python/Datasets/CV/Segmentation/PersonSeg/png/'
# save_dir_png = 'D:/A/Python/Datasets/CV/Segmentation/PersonSeg/train_white_png/'

input_dir_jpg = 'D:/A/Python/Datasets/CV/Segmentation/PersonSeg/jpg/'
save_dir_jpg = 'D:/A/Python/Datasets/CV/Segmentation/PersonSeg/simpleBG_jpg/'
input_dir_png = 'D:/A/Python/Datasets/CV/Segmentation/PersonSeg/png/'
save_dir_png = 'D:/A/Python/Datasets/CV/Segmentation/PersonSeg/simpleBG_png/'


# white_img_names = pd.read_csv('./white_imgs.csv')
white_img_names = pd.read_csv('./simpleBG_imgs.csv')
#white_img_names.head()

# copy jpg
j = 0
# 实现复制照片
for i in range(white_img_names.shape[0]):
    img_input_dir = input_dir_jpg+white_img_names.iloc[i,0]
    shutil.copy(img_input_dir, save_dir_jpg)            
    print("已经成功复制子目录照片：" + str(img_input_dir))
    j += 1
print("已经成功复制子目录：" + str(j) + "个")

    
# copy png
j = 0
# 实现复制照片
for i in range(white_img_names.shape[0]):
    png_name = white_img_names.iloc[i,0].split('.')[0]+'.png'
    img_input_dir = input_dir_png+png_name 
    shutil.copy(img_input_dir, save_dir_png)            
    print("已经成功复制子目录照片：" + str(img_input_dir))
    j += 1
print("已经成功复制子目录：" + str(j) + "个")