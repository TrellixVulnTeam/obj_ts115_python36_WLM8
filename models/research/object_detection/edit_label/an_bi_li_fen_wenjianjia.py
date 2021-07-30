import os
import shutil

train_path = "/home/db/图片/ld/img_train"
train_xml_path = "/home/db/图片/ld/xml_train"
test_path = "/home/db/图片/ld/img_test"
test_xml_path = "/home/db/图片/ld/xml_test"
proportion = 10


class FileSort(object):
    """将训练集按照设置的比例分到测试集"""
    def __init__(self, train, train_xml, test, test_xml, proportion):

        self.train = train
        self.train_xml = train_xml
        self.test = test
        self.test_xml = test_xml
        self.proportion = proportion

    def distribution_ratio(self):

        for root, folders, files in os.walk(self.train):
            index = 0
            n = 0
            for file in files:
                img_path_one = os.path.join(root, file)
                xml_path_one = os.path.join(self.train_xml, file.split('/')[-1].split('.')[-2] + ".xml")
                index += 1
                # print("第{}张图片完成".format(index))
                if index % self.proportion == 0:
                    n += 1
                    sec = xml_path_one
                    # print(sec)
                    det = os.path.join(self.test_xml, xml_path_one.split('/')[-1])
                    # print(det)
                    sec_img = os.path.join(self.train, xml_path_one.split('/')[-1].split('.')[-2] + ".jpg")
                    det_img = os.path.join(self.test, xml_path_one.split('/')[-1].split('.')[-2] + ".jpg")
                    shutil.move(sec, det)
                    shutil.move(sec_img, det_img)
                    print("已传入{}张图像".format(n))
                    print("图像是%s" % sec.split('/')[-1])


if __name__ == '__main__':
    sort = FileSort(train_path, train_xml_path, test_path, test_xml_path, proportion)
    sort.distribution_ratio()
