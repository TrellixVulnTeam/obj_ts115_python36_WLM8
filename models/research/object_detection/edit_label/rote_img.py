import os

import cv2
import numpy as np

img_dir = "/home/db/图片/picture"
result_img_dir = "/home/db/图片/img_roted"

class SetImg():
    def __init__(self,img):
        self.img = img

    def rote_90(self):
        return np.rot90(self.img)

if __name__=="__main__":
    for root,dirs,files in os.walk(img_dir):
        for file in files:
            img_path = os.path.join(img_dir,file)
            img = cv2.imread(img_path)
            set_img = SetImg(img)
            result_img = set_img.rote_90()
            result_img_path = os.path.join(result_img_dir,file)
            cv2.imwrite(result_img_path,result_img)
            print("success!")
