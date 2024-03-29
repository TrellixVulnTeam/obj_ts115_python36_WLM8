import os
import xml.etree.cElementTree as ET

# 原始ｘｍｌ文件目录
old_dir_path = '/home/db/图片/hand/hand_xml'
# 生成新ｘｍｌ文件目录
new_dir_path = '/home/db/图片/hand/hand_xml_new'

#添加需要删除的标记名称
Label_Class = ["4"]
# Label_Class = ["171_a_6_5_ymh","171_a_6_5_bj","171_a_6_5_mclhq","171_a_6_6_fdjdcsb","171_a_6_6_bx","171_a_6_9_gs","171_a_6_9_jt"]
# Label_Class = ["p"]

#添加需要修改的标签名称｛原始标签：修改后标签｝
# Edit_Label ={"cm":"hand","ay":"hand","hd":"hand","qj":"hand"}
# Edit_Label ={"nowater":"n","inwater":"i"}

class EditXmlClass:
    def __init__(self,old_dir_path,new_dir_path):
        self.old_dir_path = old_dir_path
        self.new_dir_path = new_dir_path
        self.xml_filename_list = os.listdir(self.old_dir_path)

    # 删除标签
    def delete_lable(self,label_class=[]):
        i = 0
        for index,axml in enumerate(self.xml_filename_list):
            path_xml = os.path.join(self.old_dir_path, axml)
            tree = ET.parse(path_xml)
            root = tree.getroot()

            for child in root.findall('object'):
                for name in child.iter('name'):
                    print(name.text)
                    print(label_class)
                    if name.text in label_class:
                        root.remove(child)
                        print("yes {}".format(child.text))
            tree.write(os.path.join(self.new_dir_path, axml))
            print("删除第{}张图片的多余标签".format(index+1))

    # 改变标签
    def change_label(self,Edit_Label):
        for index,axml in enumerate(self.xml_filename_list):
            path_xml = os.path.join(self.old_dir_path, axml)
            tree = ET.parse(path_xml)
            root = tree.getroot()

            for child in root.findall('object'):
                for name_node in child.iter('name'):
                    if name_node.text in Edit_Label.keys():
                        name_node.text=Edit_Label[name_node.text]
                        print("change now !")

            tree.write(os.path.join(self.new_dir_path, axml))
            print("修改第{}张图片的多余标签".format(index+1))

    def change_label(self,Edit_Label):
        for index,axml in enumerate(self.xml_filename_list):
            path_xml = os.path.join(self.old_dir_path, axml)
            tree = ET.parse(path_xml)
            root = tree.getroot()

            for child in root.findall('object'):
                for name_node in child.iter('name'):
                    if name_node.text in Edit_Label.keys():
                        name_node.text=Edit_Label[name_node.text]
                        print("change now !")

            tree.write(os.path.join(self.new_dir_path, axml))
            print("修改第{}张图片的多余标签".format(index+1))




if __name__=="__main__":
    edc = EditXmlClass(old_dir_path,new_dir_path)
    edc.delete_lable(Label_Class)
    # edc.change_label(Edit_Label)

