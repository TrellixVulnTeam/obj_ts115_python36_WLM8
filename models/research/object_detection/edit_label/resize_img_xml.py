import os

import cv2
from lxml import etree, objectify

"""
此脚本的输入为labelimg标记好的xml文件夹、原始图片文件夹
输出为缩放后的图片文件夹、新生成的图片文件夹
"""

# 原始图片的缩放比例
# resize_pix = (2080,1560)
# resize_pix = (1728,1152)
resize_pix = (640,480)
#删除没有标签的原始图片
is_delete = True
# 原始img文件夹路径
img_path = "/home/db/图片/ld/img"
# 原始xml path
xml_path = "/home/db/图片/ld/xml"

# 保存新生成的img文件夹路径
result_path_img = "/home/db/图片/ld/img_train"
# 保存新生成的xml文件夹路径
result_path_xml = "/home/db/图片/ld/xml_train"


class ResizeImageLabel():
    """

    """

    def __init__(self, img_path, xml_path, result_path_img, result_path_xml, resize_pix):
        self.img_path = img_path
        self.result_path_img = result_path_img
        self.xml_path = xml_path
        self.result_path_xml = result_path_xml
        self.src_img = cv2.imread(img_path)
        self.img = cv2.resize(self.src_img, resize_pix)
        self.src_h_scale = self.img.shape[0]/self.src_img.shape[0]
        self.src_w_scale = self.img.shape[1]/self.src_img.shape[1]
        self.resize_shape = resize_pix



    def get_points(self):
        # 获得原始xml文件中的检测框坐标
        tree = etree.parse(self.xml_path)
        points_list = []
        for bbox in tree.xpath("//bndbox"):
            name = bbox.xpath("../name")[0].text
            point_list = []
            point_location = bbox.getchildren()
            point_list.append(int(int(point_location[0].text) * self.src_w_scale))
            point_list.append(int(int(point_location[1].text) * self.src_h_scale))
            point_list.append(int(int(point_location[2].text) * self.src_w_scale))
            point_list.append(int(int(point_location[3].text) * self.src_h_scale))
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

    def creat_xml(self, img_path, xml_path, obj_points_list, img_result):
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
                E.width(self.resize_shape[0]),
                E.height(self.resize_shape[1]),
                E.depth(3)
            ),
            E.segmented(0),
        )


        for obj_point in obj_points_list:
            anno_tree.append(obj_point)

        etree.ElementTree(anno_tree).write(xml_path, pretty_print=True)
        print("xml: {} has saved!".format(xml_path))

        cv2.imwrite(img_path, img_result)
        print("img: {} has saved!".format(img_path))


    def resize_img(self):

        img_result = self.img
        img_result_path = os.path.join(self.result_path_img , self.img_path.split('/')[-1])
        # cv2.imwrite(img_result_path, img_result)
        # print("img: {} has saved!".format(img_result_path))

        # 生成新的ｘｍｌ
        point_result_list=[]
        point_one_list = self.get_points()
        for point_lc in point_one_list:
            point_one = self.create_obj_point(point_lc[0],point_lc[1],point_lc[2],point_lc[3],point_lc[4],)
            point_result_list.append(point_one)

        xml_img_path = os.path.join(self.result_path_xml,
                                    img_result_path.split('/')[-1].split('.')[-2] + '.xml')
        # xml_img_path = self.result_path_xml+
        self.creat_xml(img_result_path, xml_img_path, point_result_list, img_result)



if __name__ == "__main__":
    # 遍历文件夹内的文件，逐个裁剪
    for root, folders, files in os.walk(img_path):
        index = 0
        for file in files:
            img_path_one = os.path.join(root, file)
            xml_path_one = os.path.join(xml_path, file.split('/')[-1].split('.')[-2] + ".xml")
            if os.path.exists(xml_path_one):
                crop_image_label = ResizeImageLabel(img_path_one, xml_path_one, result_path_img, result_path_xml,resize_pix )
                crop_image_label.resize_img()
                index += 1
                print("第{}张图片缩放完成".format(index))
            if is_delete and not os.path.exists(xml_path_one):
                os.remove(img_path_one)
