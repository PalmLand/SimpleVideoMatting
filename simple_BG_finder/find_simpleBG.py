import cv2,os
import numpy as np
import pandas as pd
import find_color

 
#计算图片中每种颜色占比
def compute_colorArea(filename):
    color_areas = {}
    
    frame = cv2.imread(filename)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    color_dict = find_color.getColorList()
    for d in color_dict:
        mask = cv2.inRange(hsv,color_dict[d][0],color_dict[d][1])
        #cv2.imwrite(d+'.jpg',mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary,None,iterations=2)
        
        cnts, hiera = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        color_area = 0
        for c in cnts:
            color_area+=cv2.contourArea(c)
        color_areas[str(d)] = color_area
    
    return color_areas

#计算alpha中白色的面积，用于后续与背景图片中黑色部分面积相减
#得到背景中真实的各颜色占比
def alpha_whiteArea(filename):
    frame = cv2.imread(filename)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    color_dict = find_color.getColorList()
    #for d in color_dict:
    mask = cv2.inRange(hsv,color_dict["white"][0],color_dict["white"][1])
    #cv2.imwrite(d+'.jpg',mask)
    binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    binary = cv2.dilate(binary,None,iterations=2)

    cnts, hiera = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    color_area = 0
    for c in cnts:
        color_area+=cv2.contourArea(c)
    
    return color_area

if __name__ == '__main__':
    
    img_path = 'D:/A/Python/Datasets/CV/Segmentation/PersonSeg/background/'
    img_dirs = os.listdir(img_path)
    
    alpha_path = 'D:/A/Python/Datasets/CV/Segmentation/PersonSeg/png/'
    alpha_dirs = os.listdir(alpha_path)
        
    white_img_list = []
    for img_dir,alpha_dir in zip(img_dirs,alpha_dirs):
        white_area = alpha_whiteArea(alpha_path+alpha_dir)
        color_areas = compute_colorArea(img_path+img_dir)
        
        #得到真实的黑色区域面积
        color_areas["black"] = color_areas["black"]-white_area
        #计算各颜色占比
        sum_area = sum(color_areas.values())
        color_ratios = {}
        if sum_area==0:
            continue
        for color in color_areas:
            ratio = color_areas[color]/sum_area
            color_ratios[color] = ratio
        color_ratios = {k:v for k, v in color_ratios.items() if v>=0.05}# 过滤掉ratio小于0.15的颜色
        sum_ratio = sum(color_ratios.values())
        if sum_ratio==0:
            continue
        color_ratios_new = {}
        for color in color_ratios:
            ratio = color_ratios[color]/sum_ratio
            color_ratios_new[color] = ratio
        print(img_dir,'\t','颜色总数：',len(color_ratios_new),'\t',color_ratios_new)
        max_color = max(zip(color_ratios_new.values(), color_ratios_new.keys()))
        area = max_color[0]
        color = max_color[1]
        cond1 = color_ratios_new[color]>0.5 and len(color_ratios_new)<=4 and color=='white'
        cond2 = color_ratios_new[color]>0.5 and len(color_ratios_new)<=4 and color=='gray'
        cond3 = color_ratios_new[color]>0.5 and len(color_ratios_new)<=4 and color=='red'
        cond4 = color_ratios_new[color]>0.5 and len(color_ratios_new)<=4 and color=='red2'
        cond5 = color_ratios_new[color]>0.5 and len(color_ratios_new)<=4 and color=='orange'
        cond6 = color_ratios_new[color]>0.5 and len(color_ratios_new)<=4 and color=='yellow'
        cond7 = color_ratios_new[color]>0.5 and len(color_ratios_new)<=4 and color=='green'
        cond8 = color_ratios_new[color]>0.5 and len(color_ratios_new)<=4 and color=='cyan'
        cond9 = color_ratios_new[color]>0.5 and len(color_ratios_new)<=4 and color=='blue'
        cond10 = color_ratios_new[color]>0.5 and len(color_ratios_new)<=4 and color=='purple'
        if (cond1) or (cond2) or (cond3) or (cond4) or (cond5) or (cond6) or (cond7) or (cond8) or (cond9) or (cond10):
        #         if (color_ratios_new[color]>0.5 and len(color_ratios_new)<=2) or (color_ratios_new[color]>0.5 and len(color_ratios_new)<=2):
#         if len(color_ratios)<=2:
            white_img_list.append(img_dir)
    
    # 保存文件
    white_img_npy = np.array(white_img_list)
    np.save('simpleBG_imgs.npy',white_img_npy)
    '''
    # 读取
    a=np.load('simpleBG_imgs.npy')
    a=a.tolist()
    '''
    
    white_img_csv = pd.DataFrame(white_img_list) 
    white_img_csv.to_csv('simpleBG_imgs.csv',index=0)