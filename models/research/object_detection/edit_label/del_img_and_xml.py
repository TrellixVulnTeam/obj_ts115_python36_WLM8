import os
import shutil

from lxml import etree

img_path = "/media/m/涉密/5720项目图片及标签/弹射座椅/2019/img"
xml_path = "/media/m/涉密/5720项目图片及标签/弹射座椅/2019/img_xml"
img_new = "/media/m/涉密/5720项目图片及标签/弹射座椅/2019/img_new"
xml_new = "/media/m/涉密/5720项目图片及标签/弹射座椅/2019/img_xml_new"


class DelImgLabel():

    def __init__(self, img_path, xml_path, img_new, xml_new):
        self.img_path = img_path
        self.xml_path = xml_path
        self.img_new = img_new
        self.xml_new = xml_new

    def get_obj(self):
        tree = etree.parse(self.xml_path)
        list1 = []
        for obj in tree.xpath("//*"):
            obj = str(obj)
            print(obj)
            list1.append(obj)

            print(list1)
            if "object" in obj:
                sec = self.xml_path
                print(sec)
                det = os.path.join(self.xml_new, self.xml_path.split('/')[-1])
                print(det)
                sec_img = os.path.join(self.img_path, self.xml_path.split('/')[-1].split('.')[-2] + ".jpg")
                det_img = os.path.join(self.img_new, self.xml_path.split('/')[-1].split('.')[-2] + ".jpg")
                shutil.copy(sec, det)
                shutil.copy(sec_img, det_img)
                print("1")


if __name__ == '__main__':
    for root, folders, files in os.walk(img_path):
        index = 0
        for file in files:
            img_path_one = os.path.join(root, file)
            xml_path_one = os.path.join(xml_path, file.split('/')[-1].split('.')[-2] + ".xml")
            del_image_label = DelImgLabel(img_path, xml_path_one, img_new, xml_new)
            del_image_label.get_obj()
            index += 1
            print("第{}张图片完成".format(index))


