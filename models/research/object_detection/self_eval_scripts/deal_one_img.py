import re

import cv2
import tensorflow as tf

from object_detection.self_eval_scripts.eval_img_class.load_pb_model import LoadPbModel

# from django_tensorflow_server.settings import BASE_DIR

# img_path = os.path.join(BASE_DIR, "test_img/kougai.jpg")
img_path = "/media/db/WLZ_Secret_db/备份数据集(非常重要)/5720/5720座椅/test/img/1624695064024.jpeg"
# img_path = os.path.join(BASE_DIR, "test_img/toubu22.jpg")

# model_path = saved_model_dir = os.path.join(BASE_DIR, "pb_model/fangfei/kougai/saved_model")
model_path = saved_model_dir = "saved_model"
# resize_shape = (640, 480)
resize_shape = (640, 480)
repeat_iou = 0.5
show_rate = 0.5

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True
g1 = tf.Graph()
sess = tf.Session(config=config, graph=g1)
meta_graph_def_sig = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)

label_path = "pascal_label_map.pbtxt"


class KouGaiEval():
    def __init__(self, sess=sess,label_path=label_path):
        self.load_pb_model = LoadPbModel(sess)

        self.label_dict = {}
        with open(label_path, 'r+') as f:
            lable_str = f.read()
            lable_list = re.findall(".*name: '(.*)'", lable_str)
            for index, lable_one in enumerate(lable_list):
                self.label_dict[index + 1] = lable_one

    def get_detect_result(self, img_path, resize_shape=resize_shape):
        img_list = self.load_pb_model.read_img(img_path, resize_shape)
        y = self.load_pb_model.eval_img_data_list(img_list)
        result_list = self.load_pb_model.get_img_result_list(y, repeat_iou=repeat_iou, show_rate=show_rate)
        result_list_wrong = []
        # for x in result_list:
        #     if x[4] in [1.0,2.0,3.0,4.0]:
        #         # result_list_wrong.append(x)
        #         if x[4]==1.0 and x[5]>0.8:
        #             result_list_wrong.append(x)
        #         if x[4]==2.0 and x[5]>0.8:
        #             result_list_wrong.append(x)
        #         if x[4]==3.0 and x[5]>0.4:
        #             result_list_wrong.append(x)
        #         if x[4]==4.0 and x[5]>0.5:
        #             result_list_wrong.append(x)

        # a = 3
        # img_result = self.load_pb_model.draw_boxes(result_list, img_list[0],self.label_dict)
        img_result = self.load_pb_model.draw_boxes(result_list, img_list[0],self.label_dict)

        cv2.imshow("img_result", img_result)
        cv2.waitKey(1)
        return result_list_wrong


if __name__ == "__main__":
    load_pb_model = KouGaiEval()
    cap = cv2.VideoCapture(0)
    cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(3, 480)
    cap.set(4, 640)
    while True:
        ret, frame = cap.read()
        # frame = cv2.imread("/home/db/图片/座椅测试图片/1624694645697.jpeg")
        img_list = load_pb_model.get_detect_result(frame)
    # img_result = load_pb_model.draw_boxes(img_list, img_path)
    # cv2.imshow("img_result", img_result)
    # cv2.waitKey(0)
    # a = 222
