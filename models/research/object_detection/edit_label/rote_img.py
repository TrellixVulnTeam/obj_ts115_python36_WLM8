import os

import cv2
import numpy as np

img_dir = "/media/db/WLZ_Secret_db/备份数据集(非常重要)/成飞项目图片及标签/孔洞识别/Camera"
result_img_dir = "/media/db/WLZ_Secret_db/备份数据集(非常重要)/成飞项目图片及标签/孔洞识别/img"

class SetImg():
    def __init__(self,img):
        self.img = img

    def rote_90(self):
        return np.rot90(self.img)

if __name__=="__main__":
    for root,dirs,files in os.walk(img_dir):
        for index,file in enumerate(files):
            img_path = os.path.join(img_dir,file)
            img = cv2.imread(img_path)
            if img.shape[0]>img.shape[1]:
            # print(img.shape)
                set_img = SetImg(img)
                result_img = set_img.rote_90()
            else:
                result_img = img
            result_img_path = os.path.join(result_img_dir,file)
            cv2.imwrite(result_img_path,result_img)
            print("success {}!".format(index))
