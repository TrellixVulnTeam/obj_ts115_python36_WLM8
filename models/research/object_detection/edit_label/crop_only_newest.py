import math
import os

import cv2
from lxml import etree, objectify

"""
此脚本的输入为labelimg标记好的xml文件夹、原始图片文件夹
输出为裁剪后的图片文件夹、新生成的图片文件夹
"""

# 原始图片的缩放比例

resize_shape = 1.0
# resize_shape = 0.50
# resize_shape = 0.33
# resize_shape = 0.25

# 裁剪的大小
# （宽，长 ）
# crop_size = (640, 640)
crop_size = (480, 854)
# 重合的边界
# border = 110
border = 250
# boxes重合IOU阈值
box_iou = 0.85
# box_iou = 1.0
# 是否保留没有目标的裁剪框
# saved_only_boxes = True
saved_only_boxes = False
# 测试数据集的比例
# test_rate = 0.1
test_rate = 0

# 原始img文件夹路径
img_path = "/media/db/B47A-50F6/20201118孔/bianhuan/img"
# 原始xml文件夹路径
xml_path = "/media/db/B47A-50F6/20201118孔/bianhuan/xml"

# 保存裁剪img和xml文件的根路径
crop_root = "/media/db/B47A-50F6/20201118孔/bianhuan/croped"
# 保存新生成的训练img文件夹路径
train_path_img = os.path.join(crop_root, "train_img")
# 保存新生成的训练img文件夹路径
train_path_xml = os.path.join(crop_root, "train_xml")
# 保存新生成的测试img文件夹路径
test_path_img = os.path.join(crop_root, "test_img")
# 保存新生成的测试xml文件夹路径
test_path_xml = os.path.join(crop_root, "test_xml")


class CropImageLabel():
    """
    把大分辨率的图片裁剪成小分辨率的图片，同时生成对应的标签文件
    """

    def __init__(self, img_path, xml_path, crop_size, border, resize_shape):
        self.img_path = img_path
        # self.result_path_img = result_path_img
        self.crop_size = crop_size
        self.xml_path = xml_path
        # self.result_path_xml = result_path_xml
        self.border = border
        self.img = cv2.imread(img_path)
        self.resize_shape = resize_shape
        self.img = cv2.resize(self.img,
                              (int(self.img.shape[1] * self.resize_shape), int(self.img.shape[0] * self.resize_shape)))

    def get_points(self):
        # 获得原始xml文件中的检测框坐标
        tree = etree.parse(self.xml_path)
        points_list = []
        for bbox in tree.xpath("//bndbox"):
            name = bbox.xpath("../name")[0].text
            point_list = []
            for points in bbox.getchildren():
                point_list.append(int(int(points.text) * self.resize_shape))
            point_list.append(name)

            points_list.append(point_list)
        # print(points_list)
        return points_list

    def create_obj_point(self, xmin, ymin, xmax, ymax, name):
        # 创建新的object节点
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E2.name(name),
            E2.pose("Unspecified"),
            E2.truncated(0),
            E2.difficult(0),
            E2.bndbox(
                E2.xmin(xmin),
                E2.ymin(ymin),
                E2.xmax(xmax),
                E2.ymax(ymax)
            ),
        )
        return anno_tree2

    def creat_xml(self, img_path, xml_path, obj_points_list, img_result, delete_none_boxes):
        # 创建新的xml文件
        E = objectify.ElementMaker(annotate=False)
        anno_tree = E.annotation(
            E.folder(img_path.split('/')[-2]),
            E.filename(img_path.split('/')[-1]),
            E.path(img_path),
            E.source(
                E.database('Unknown'),
            ),
            E.size(
                E.width(self.crop_size[0]),
                E.height(self.crop_size[1]),
                E.depth(3)
            ),
            E.segmented(0),
        )

        # anno_tree2 = self.create_obj_point(1, 1, 1, 1)
        # 保留if语句会只保留至少有一个目标的裁剪区域，删除if语句会保留所有裁剪的图片
        if delete_none_boxes:
            if obj_points_list:
                for obj_point in obj_points_list:
                    anno_tree.append(obj_point)

                etree.ElementTree(anno_tree).write(xml_path, pretty_print=True)
                print("xml: {} has saved!".format(xml_path))

                cv2.imwrite(img_path, img_result)
                print("img: {} has saved!".format(img_path))
        else:
            for obj_point in obj_points_list:
                anno_tree.append(obj_point)

            etree.ElementTree(anno_tree).write(xml_path, pretty_print=True)
            print("xml: {} has saved!".format(xml_path))

            cv2.imwrite(img_path, img_result)
            print("img: {} has saved!".format(img_path))

    def crop_img(self, box_iou, result_path_img, result_path_xml, ):
        # 裁剪图片
        h, w = self.img.shape[:2]
        h_num = math.floor((h - self.crop_size[0]) / (self.crop_size[0] - self.border)) + 2
        w_num = math.floor((w - self.crop_size[1]) / (self.crop_size[1] - self.border)) + 2

        a1 = h_num * self.crop_size[0] + self.border - h
        a2 = w_num * self.crop_size[1] + self.border - w
        img = cv2.copyMakeBorder(self.img, 0, h_num * self.crop_size[0] + self.border - h, 0,
                                 w_num * self.crop_size[1] + self.border - w, cv2.BORDER_CONSTANT,
                                 value=[255, 255, 255])

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
                                               self.img_path.split('/')[-1].split('.')[0]+"_{}".format(str(self.resize_shape).replace(".","__"))+ "_{}".format(
                                                   str(x_l) + "_" + str(y_l)) + ".jpg")

                # cv2.imwrite(img_result_path, img_result)
                # break
                # print("img: {} has saved!".format(img_result_path))

                # 生成新的ｘｍｌd
                points_list = self.get_points()
                point_one_list = []

                for point in points_list:
                    if ((point[0] > y_min and point[0] < y_max) and (point[1] > x_min and point[1] < x_max)) or (
                            (point[2] > y_min and point[2] < y_max) and (point[3] > x_min and point[3] < x_max)):

                        # 原始大图片box坐标(y1,x1,y2,x2)对应在裁剪后图片中的坐标
                        py1 = point[0] - y_min
                        px1 = point[1] - x_min
                        py2 = point[2] - y_min
                        px2 = point[3] - x_min

                        # 如果裁剪到box的边缘,则box的坐标取边界值
                        py1 = max(py1, 0)
                        px1 = max(px1, 0)
                        py2 = min(py2, self.crop_size[1])
                        px2 = min(px2, self.crop_size[0])

                        # 原始box的面积
                        area_large = abs(point[0] - point[2]) * abs(point[1] - point[3])
                        # 　裁剪后box的面积
                        area_small = abs(py1 - py2) * abs(px1 - px2)
                        # 面积重叠概率
                        iou_iou = area_small / area_large
                        # 若重叠面积大于预设的阈值则保留
                        if iou_iou >= box_iou:
                            point_one = self.create_obj_point(py1, px1, py2, px2, point[4])
                            point_one_list.append(point_one)

                # a = img_result_path.split('/')[-1].split
                xml_img_path = os.path.join(result_path_xml,
                                            img_result_path.split('/')[-1].split('.')[-2] + '.xml')
                # xml_img_path = self.result_path_xml+
                self.creat_xml(img_result_path, xml_img_path, point_one_list, img_result, saved_only_boxes)

                # print(x_min, x_max, y_min, y_max)
                # img_result = img[x_min:x_max, y_min:y_max]
                # img_result_path = self.result_path_img + self.img_path.split('/')[-1].split('.')[0] + "_{}".format(
                #     str(x_l) + "_" + str(y_l)) + ".jpg"
                # cv2.imwrite(img_result_path, img_result)
                # print("img: {} has saved!".format(img_result_path))


if __name__ == "__main__":
    for file_path in [crop_root,train_path_img,train_path_xml,test_path_img,test_path_xml]:
        os.makedirs(file_path,exist_ok=True)
    # 遍历文件夹内的文件，逐个裁剪
    for root, folders, files in os.walk(img_path):
        for index, file in enumerate(files):
            img_path_one = os.path.join(root, file)
            xml_path_one = os.path.join(xml_path, file.split('/')[-1].split('.')[-2] + ".xml")
            # print(img_path_one)
            # print(xml_path_one)
            result_path_img=train_path_img
            result_path_xml=train_path_xml

            if test_rate!=0:
                if not index%int(1/test_rate):
                    # print("tttttttttttttttttttttttttttttttttttt")
                    result_path_img = test_path_img
                    result_path_xml = test_path_xml

            crop_image_label = CropImageLabel(img_path_one, xml_path_one, crop_size,
                                              border, resize_shape)
            crop_image_label.crop_img(box_iou,result_path_img,result_path_xml)
            index += 1
            print("第{}张图片裁剪完成".format(index))
