import cv2
import numpy as np
import tensorflow as tf

# 图片大小
img_shape = (480, 480)
resize_shape = (480, 480)
# 摄像头序号
camera_id = 0
#测试图像路径
img_path = "/media/db/WLZ_Secret_db/备份数据集(非常重要)/5720/5720座椅/20210317_all/座椅/test/20190613_000025.jpg"
# 测试文件路径
tflite_path = "/home/db/文档/hz.tflite"
score = 0.6
# 标签文件,根据打标签时候的情况修改
label_dict = {'1': '2',
              '2': '3',
              '3': '4',
              '4': '5',
              '5': '6',
              '6': '7',
              '7': 's1',
              '8': 's2',
              '9': 's3',
              '10': 's4',
              '11': 's5',
              }
# label_dict = {'1': '1',
#               '2': '2',
#               '3': '3',
#               '4': '4',
#               '5': '5',
#               '6': '6',
#               '7': '7',
#               '8': '8',
#               '9': 'l',
#               '10': 's',
#               '11': '2l',
#               '12': '2s',
#               '13': '0',
#               '14': '1b',
#               '15': '2b',
#               '16': '3b',
#               '17': '4b',
#               '18': '4c',
#               '19': '4d',
#               '20': '5b',
#               '21': '6b',
#               '22': '78',
#               '23': '0'
#               }



class GetOneImg():
    def __init__(self, camera_id, resize_shape):
        self.cap = cv2.VideoCapture(camera_id)
        self.resize_shape = resize_shape
        self.cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        # self.cap.set(3, 480)
        # self.cap.set(4, 480)

    def get_one_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, self.resize_shape)
        return frame

    def get_one_img(self,frame):
        frame = cv2.resize(frame, self.resize_shape)
        return frame

    def prepare_input_data(self, frame):
        img = cv2.resize(frame, self.resize_shape)
        img = img[..., ::-1]
        img_array = np.array(img, dtype=float)
        img_array = img_array[np.newaxis, :, :, :]
        input_data = np.array(img_array, dtype=np.float32)
        input_data = ((input_data / 127.5) - 1)
        # input_data = (input_data / 255.0)
        return input_data


class DetectImg():
    def __init__(self, tflite_path, score):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.score = score

    def is_detected(self, input_data):
        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
        except RuntimeError:
            print("RuntimeError!")
            return False
        else:
            print("pass")
            return True

    def detect(self, frame):
        boxes_list = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        class_list = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        score_list = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        num_list = self.interpreter.get_tensor(self.output_details[3]['index'])[0]

        # print("box_list: {}".format(boxes_list))
        # print("class_list: {}".format(class_list))
        # print("score_list: {}".format(score_list))
        # print("num_list: {}".format(num_list))

        result_list = []

        for i in range(len(boxes_list)):
            if score_list[i] > self.score:
                # print("----------------------")
                # print(score_list[i])
                one_boxes = [float(boxes_list[i][0]), float(boxes_list[i][1]), float(boxes_list[i][2]),
                             float(boxes_list[i][3]),
                             float(score_list[i]), int(class_list[i])]
                # print(one_boxes)
                result_list.append(one_boxes)

        for box in result_list:
            point_1 = (int(box[1] * img_shape[0]), int(box[0] * img_shape[1]))
            point_2 = (int(box[3] * img_shape[0]), int(box[2] * img_shape[1]))
            # print(point_1, point_2)
            # cv2.rectangle(img, (240, 0), (480, 375), (0, 255, 0), 2)
            cv2.rectangle(frame, point_1, point_2, (255, 0, 0), 1)
            # cv2.imshow('sdfa',img)
            # cv2.waitKey(0)
            str_txt = label_dict[str(box[5] + 1)]
            score_one = str(box[4])[:4]
            # print("str_txt{}".format(str_txt))
            cv2.putText(frame, score_one, point_2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(frame, str_txt, point_1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        if not result_list:
            cv2.putText(frame, "box none !", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

        return frame


if __name__ == "__main__":
    get_one_img = GetOneImg(camera_id, resize_shape)
    detect_img = DetectImg(tflite_path, score)

    # frame = cv2.imread(img_path)
    # img_src_shape = (frame.shape[1],frame.shape[0])
    # frame = get_one_img.get_one_img(frame)
    # input_data = get_one_img.prepare_input_data(frame)
    # is_detect = detect_img.is_detected(input_data)
    #
    # if is_detect:
    #     result_img = detect_img.detect(frame)
    #     # cv2.imshow("saf",result_img)
    #     result_img = cv2.resize(result_img, img_src_shape)
    #     cv2.imshow("detect_result", result_img)
    # else:
    #     cv2.putText(frame, "model error !", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    #     cv2.imshow("detect_result", frame)
    # cv2.waitKey(0)

    # 调用摄像头
    while True:
        frame = get_one_img.get_one_frame()
        input_data = get_one_img.prepare_input_data(frame)
        is_detect = detect_img.is_detected(input_data)
        if is_detect:
            result_img = detect_img.detect(frame)
            cv2.imshow("detect_result", result_img)
        else:
            cv2.putText(frame, "model error !", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
            cv2.imshow("detect_result", frame)
        cv2.waitKey(10)
