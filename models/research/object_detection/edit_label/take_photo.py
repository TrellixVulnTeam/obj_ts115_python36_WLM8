import os
import time

import cv2

img_save_dir = "./img"
if not os.path.exists(img_save_dir):
    os.mkdir(img_save_dir)

cap = cv2.VideoCapture(0)
cap.set(6,cv2.VideoWriter.fourcc('M','J','P','G'))
cap.set(3,640) #设置分辨率
cap.set(4,480)
fps =cap.get(cv2.CAP_PROP_FPS)
# print(str(time.time()))
index = 0

while True:
    # get a frame
    ret, frame = cap.read()

    input = cv2.waitKey(1) & 0xFF

    if input == ord('p'):
        index +=1
        img_file_path = os.path.join(img_save_dir,str(time.time())+".png")
        cv2.imwrite(img_file_path,frame)
        print("第{}张图片拍摄完成，保存到{}".format(index,img_file_path))
        cv2.imshow("capture", frame)
        cv2.waitKey(100)
    else:
        cv2.imshow("capture", frame)


