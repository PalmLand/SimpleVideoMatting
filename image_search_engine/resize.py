import cv2,os
import numpy as np
import pandas as pd

# queriy_path = './whiteBG/'
# save_path = './whiteBG_resize/'
queriy_path = './whitenoBG/'
save_path = './whitenoBG_resize/'
queriy_dirs = os.listdir(queriy_path)
for queriy_dir in queriy_dirs:
    img = cv2.imread(queriy_path+queriy_dir)
    img = cv2.resize(img,(320,320))
    cv2.imwrite(save_path+queriy_dir,img)