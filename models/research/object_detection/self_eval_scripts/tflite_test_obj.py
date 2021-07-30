import cv2
import numpy as np
import tensorflow as tf

# label_dict = {'1': 'lsd', }
# label_dict = {'1': '1',
#               '2': '2',
#               '3': '3',
#               '4': '4',
#               '5': '5',
#               '6': '6',
#               '7': '7',
#               '8': '8',
#               '9': '1',
#               '10': 's',
#               '11': '2l',
#               '12': '2s',
#               '13': '1b',
#               '14': '2b',
#               '15': '3b',
#               '16': '4b',
#               '17': '4c',
#               '18': '4d',
#               '19': '5b',
#               '20': '6b',
#               '21': '78',
#               '22': '0'
#               }
# img_shape = (640, 480)
# camera_id = 0
# resize_shape = (640, 480)
# tflite_path = "/home/db/图片/81969.tflite"
# score = 0.5
# label_dict = {'1': '2',
#               '2': '3',
#               '3': '4',
#               '4': '5',
#               '5': '6',
#               '6': '7',
#               '7': 's0',
#               '8': 's1',
#               '9': 's2',
#               '10': 's3',
#               '11': 's4',
#               '12': '2s',
#               '13': '1b',
#               '14': '2b',
#               '15': '3b',
#               '16': '4b',
#               '17': '4c',
#               '18': '4d',
#               '19': '5b',
#               '20': '6b',
#               '21': '78',
#               '22': '0'
#               }
label_dict = {'1': 'chumo',
              '2': 'huangdong',
              '3': 'shoudong',
              '4': 'xuanzhuan',
              '5': 'anya',
              }
img_shape = (640, 480)
camera_id = 0
resize_shape = (640, 480)
tflite_path = "84991.tflite"
score = 0.5


# label_dict = {'1':'k','2':'y'}
# label_dict = {'1':'r','2':'f'}
# label_dict = {'1':'0','2':'1','3':'2','4':'3','5':'4','6':'5','7':'6','8':'7','9':'8','10':'9'}
# label_dict = {'1': '0', '2': '1', '3': '2', '4': '3', '5': '4', '6': '5', '7': '6', '8': '7', '9': '8', '10': '9',
#               '11': 'A', '12': 'B', '13': 'C', '14': 'D', '15': 'E', '16': 'F', '17': 'G', '18': 'H', '19': 'I',
#               '20': 'J',
#               '21': 'K', '22': 'L', '23': 'M', '24': 'N', '25': 'O', '26': 'P', '27': 'Q', '28': 'R', '29': 'S',
#               '30': 'T',
#               '31': 'U', '32': 'V', '33': 'W', '34': 'X', '35': 'Y', '36': 'Z',
#               '37': 'a', '38': 'b', '39': 'c', '40': 'd', '41': 'e', '42': 'f', '43': 'g', '44': 'h', '45': 'i',
#               '46': 'j',
#               '47': 'k', '48': 'l', '49': 'm', '50': 'n', '51': 'o', '52': 'p', '53': 'q', '54': 'r', '55': 's',
#               '56': 't',
#               '57': 'u', '58': 'v', '59': 'w', '60': 'x', '61': 'y', '62': 'x'
#               }


class GetOneImg():
    def __init__(self, camera_id, resize_shape):
        self.cap = cv2.VideoCapture(camera_id)
        self.resize_shape = resize_shape
        self.cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(3, 480)
        self.cap.set(4, 640)

    def get_one_frame(self):
        ret, frame = self.cap.read()
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

    def is_detected(self,input_data):
        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
        except RuntimeError:
            print("RuntimeError!")
            return False
        else:
            print("pass")
            return True

    def detect(self,frame):
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
            cv2.putText(frame, "box none !", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

        return frame


if __name__ == "__main__":
    get_one_img = GetOneImg(camera_id, resize_shape)
    detect_img = DetectImg(tflite_path, score)

    while True:
        frame = get_one_img.get_one_frame()
        input_data = get_one_img.prepare_input_data(frame)
        is_detect = detect_img.is_detected(input_data)
        if is_detect:
            result_img = detect_img.detect(frame)
            cv2.imshow("detect_result", result_img)
        else:
            cv2.putText(frame, "model error !", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
            cv2.imshow("detect_result",frame)
        cv2.waitKey(10)

