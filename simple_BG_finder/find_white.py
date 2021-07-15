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
        
        max_color = max(zip(color_areas.values(), color_areas.keys()))
        area = max_color[0]
        color = max_color[1]
        if color == 'white':
            print(img_dir)
            white_img_list.append(img_dir)
            
    # 保存文件
    white_img_npy = np.array(white_img_list)
    np.save('white_imgs.npy',white_img_npy)
    '''
    # 读取
    a=np.load('white_imgs.npy')
    a=a.tolist()
    '''
    
    white_img_csv = pd.DataFrame(white_img_list) 
    white_img_csv.to_csv('white_imgs.csv',index=0)