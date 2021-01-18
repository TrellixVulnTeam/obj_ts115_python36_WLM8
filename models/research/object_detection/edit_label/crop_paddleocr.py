import math
import os

import cv2

"""
此脚本的输入为labelimg标记好的xml文件夹、原始图片文件夹
输出为裁剪后的图片文件夹、新生成的图片文件夹
"""

# 原始图片的缩放比例

resize_shape = 1.0
# resize_shape = 0.8
# resize_shape = 0.33
# resize_shape = 0.25
# resize_shape = 0.1

# 裁剪的大小
# （宽，长 ）
crop_size = (32, 150)
# crop_size = (2000, 2000)
# crop_size = (480, 854)
# 重合的边界
# border = 50
# border = 110
border = 5
# boxes重合IOU阈值
box_iou = 0.0
# box_iou = 0.85
# box_iou = 1.0
# 是否保留没有目标的裁剪框
saved_only_boxes = True
# saved_only_boxes = False
# 测试数据集的比例
# test_rate = 0.1
test_rate = 0

# 原始img文件夹路径
img_path = "/home/db/视频/steal_img"

# 保存裁剪img和xml文件的根路径
crop_root = "/home/db/视频/croped_steal_img"
# 保存新生成的训练img文件夹路径
train_path_img = os.path.join(crop_root, "train_img")
# 保存新生成的测试img文件夹路径
test_path_img = os.path.join(crop_root, "test_img")



class CropImageLabel():
    """
    把大分辨率的图片裁剪成小分辨率的图片，同时生成对应的标签文件
    """

    def __init__(self, img_path, crop_size, border, resize_shape):
        self.img_path = img_path
        # self.result_path_img = result_path_img
        self.crop_size = crop_size
        # self.result_path_xml = result_path_xml
        self.border = border
        self.img = cv2.imread(img_path)
        self.resize_shape = resize_shape
        self.img = cv2.resize(self.img,
                              (int(self.img.shape[1] * self.resize_shape), int(self.img.shape[0] * self.resize_shape)))


    def crop_img(self, result_path_img):
        # 裁剪图片
        h, w = self.img.shape[:2]

        h_num = math.ceil((h - self.crop_size[0]) / (self.crop_size[0] - self.border)) + 1
        w_num = math.ceil((w - self.crop_size[1]) / (self.crop_size[1] - self.border)) + 1

        # 需要添加的边界
        b_h = self.crop_size[0] - ((h-self.crop_size[0])%(self.crop_size[0]-border))-border
        b_w = self.crop_size[1] - ((w - self.crop_size[1]) % (self.crop_size[1] - border))-border


        img = cv2.copyMakeBorder(self.img, 0, b_h, 0,
                                 b_w, cv2.BORDER_WRAP)

        # cv2.imwrite("1234.jpg", img)

        h, w = img.shape[:2]

        # print(h_num, w_num)
        for x_l in range(h_num):
            for y_l in range(w_num):
                # print(x_l, y_l)
                x_min = x_l * (self.crop_size[0] - self.border)
                x_max = x_min + self.crop_size[0]
                y_min = y_l * (self.crop_size[1] - self.border)
                y_max = y_min + self.crop_size[1]

                if x_max >= h:
                    x_max = h
                if y_max >= w:
                    y_max = w

                img_result = img[x_min:x_max, y_min:y_max]
                # img_result_path = os.path.join(result_path_img,
                #                                self.img_path.split('/')[-1].split('.')[0] + "_{}".format(
                #                                    str(x_l) + "_" + str(y_l)) + ".jpg")
                img_result_path = os.path.join(result_path_img,
                                               self.img_path.split('/')[-1].split('.')[0] + "_{}".format(
                                                   str(self.resize_shape).replace(".", "__")) + "_{}".format(
                                                   str(x_l) + "_" + str(y_l)) + ".jpg")

                cv2.imwrite(img_result_path, img_result)
                # break
                print("img: {} has saved!".format(img_result_path))

if __name__ == "__main__":
    for file_path in [crop_root, train_path_img,test_path_img]:
        os.makedirs(file_path, exist_ok=True)
    # 遍历文件夹内的文件，逐个裁剪
    for root, folders, files in os.walk(img_path):
        for index, file in enumerate(files):
            img_path_one = os.path.join(root, file)
            # xml_path_one = os.path.join(xml_path, file.split('/')[-1].split('.')[-2] + ".xml")
            # print(img_path_one)
            # print(xml_path_one)
            result_path_img = train_path_img

            if test_rate != 0:
                if not index % int(1 / test_rate):
                    # print("tttttttttttttttttttttttttttttttttttt")
                    result_path_img = test_path_img

            crop_image_label = CropImageLabel(img_path_one, crop_size,
                                              border, resize_shape)
            crop_image_label.crop_img(result_path_img)
            index += 1
            print("第{}张图片裁剪完成".format(index))
