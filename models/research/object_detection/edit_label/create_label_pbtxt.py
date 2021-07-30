import os

from lxml import etree

"""
此脚本的输入为labelimg标记好的xml文件夹,
输出为标签文件txt
"""

# xml path
xml_path = "/home/db/图片/ld/xml"
label_txt_path = "/home/db/图片/ld/xml/pascal_label_map.pbtxt"


def get_points(xml_path):
    # 获得原始xml文件中的检测框坐标
    tree = etree.parse(xml_path)
    label_one_list = []
    if tree.xpath("//object"):
        # 查找所有label名字
        for bbox in tree.xpath("//object"):
            name = bbox.xpath("//name")
        for label_name in name:
            label_one_list.append(label_name.text)
        return label_one_list
    if not tree.xpath("//bndbox"):
        print(xml_path)
        return []


if __name__ == "__main__":
    # 遍历文件夹内的文件，逐个裁剪
    lable_list = []
    label_dict = {'7': 0, 's': 0, '0': 0, '78': 0, '8': 0, 'l': 0, '1': 0, '1b': 0, '6': 0, '6b': 0, '2': 0, '3': 0,
                  '3b':0, '5':0, '5b':0, '4':0, '4b':0, '2l':0, '2b':0, '4d':0, '2s':0, '4c':0}
    for root, folders, files in os.walk(xml_path):
        index = 0
        for file in files:
            xml_path_one = os.path.join(root, file)
            label_one_list = get_points(xml_path_one)
            for label_one in label_one_list:
                if label_one not in lable_list and label_one != []:
                    # label_dict[label_one] =
                    lable_list.append(label_one)

    print(lable_list)

    with open(label_txt_path, 'w') as f:
        label_list = sorted(lable_list)
        for index, x in enumerate(lable_list):
            str_label = "item {\n" + "id: {}\n".format(index + 1) + "name: '{}'\n".format(x) + "}\n"
            f.write(str_label)

    print(lable_list)
