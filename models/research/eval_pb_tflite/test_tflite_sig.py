import cv2
import numpy as np
import tensorflow as tf
from del_cf import Del_Cf

# test_dir_path = '/media/sucom/K_480/test_img'
# test_img_path = '/media/sucom/K_480/图片/20200618/du/20200618_163757.jpg'
# test_img_path = '/home/sucom/Downloads/test/1427187363.jpg'
# img = cv2.imread(test_img_path)
# img = cv2.resize(img, (640, 480))
# for files in os.walk(test_dir_path):
#     for file in files
#         print(os.path.join(test_dir_path, file))
# img = cv2.imread(os.path.join(test_dir_path,file))
# for file in os.listdir(test_dir_path):
#     img = cv2.imread(os.path.join(test_dir_path, file))
# label_dict = {'1.0':'k','2.0':'g','3.0':'l','4.0':'r','5.0':'m'}

label_dict = {'1':'lsd',}
# label_dict = {'1':'k','2':'y'}
# label_dict = {'1':'r','2':'f'}



# img = cv2.imread('8.jpg')
def deal_one_img(frame, img_name):
    # img = cv2.imread('26.jpg')
    # img = cv2.resize(frame,(480,480))
    img = cv2.resize(frame, (640, 480))
    img = img[..., ::-1]
    interpreter = tf.lite.Interpreter(
        # model_path="/home/sucom/Documents/ocr/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/frozen/detect_ocr_229889.tflite")
        model_path="/home/sucom/Documents/lsd_480_20201020/frozen/detect_lsd.tflite")

    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details)
    # print(output_details)

    # Test model on random input data.
    input_shape = input_details[0]['shape']

    img_array = np.array(img, dtype=float)
    img_array = img_array[np.newaxis, :, :, :]
    # print(img_array.shape)
    # cv2.waitKey(0)
    # Load TFLite model and allocate tensors.
    # interpreter = tf.lite.Interpreter(model_path="/home/db/视频/models-master/research/object_detection/detect.tflite")
    # interpreter.allocate_tensors()
    #
    # # Get input and output tensors.
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    # # print(input_details)
    # # print(output_details)
    #
    # # Test model on random input data.
    # input_shape = input_details[0]['shape']
    # print(input_shape)
    # print(output_details[0]['shape'])
    # print(output_details[1]['shape'])
    # print(output_details[2]['shape'])
    # print(output_details[3]['shape'])
    # print("+++++++++++++++++++++++++++")
    input_data = np.array(img_array, dtype=np.float32)
    # input_image =（1.0 / 255.0）*input_image
    input_data = ((input_data / 127.5) - 1)
    # input_data = (input_data / 255.0)

    # input_data = (1.0/255.0)*input_data

    # print(type(input_data[0][0][0][0]))
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # interpreter.invoke()

    while True:
        try:
            interpreter.invoke()
        except RuntimeError:
            print("RuntimeError")
        else:
            break


    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    boxes_list = interpreter.get_tensor(output_details[0]['index'])[0]
    class_list = interpreter.get_tensor(output_details[1]['index'])[0]
    score_list = interpreter.get_tensor(output_details[2]['index'])[0]
    num_list = interpreter.get_tensor(output_details[3]['index'])[0]

    print(boxes_list)
    print(class_list)
    print(score_list)
    print(num_list)

    result_list = []

    for i in range(len(boxes_list)):
        if score_list[i] > 0.7:
            print("----------------------")
            print(score_list[i])
            one_boxes = [float(boxes_list[i][0]), float(boxes_list[i][1]), float(boxes_list[i][2]),
                         float(boxes_list[i][3]),
                         float(score_list[i]), int(class_list[i])]
            print(one_boxes)
            result_list.append(one_boxes)

    del_cf = Del_Cf()
    result_list = del_cf.del_iou_boxes(
        [[box_one[1], box_one[0], box_one[3], box_one[2], box_one[4], box_one[5]] for box_one in result_list])
    result_list = [[box_one[1], box_one[0], box_one[3], box_one[2], box_one[4], box_one[5]] for box_one in result_list]

    for box in result_list:
        point_1 = (int(box[1] * 640), int(box[0] * 480))
        point_2 = (int(box[3] * 640), int(box[2] * 480))
        print(point_1, point_2)
        # cv2.rectangle(img, (240, 0), (480, 375), (0, 255, 0), 2)
        cv2.rectangle(frame, point_1, point_2, (255, 0, 0), 1)
        # cv2.imshow('sdfa',img)
        # cv2.waitKey(0)
        str_txt = label_dict[str(box[5] + 1)]
        score_one = str(box[4])[:4]
        # print("str_txt{}".format(str_txt))
        cv2.putText(frame, score_one, point_2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, str_txt, point_1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

    return frame

# class GetOneImg():
#     def __init__(self,cap,resize_shape,):
#         self.cap = cap
#         self.resize_shape = resize_shape
#
#     def get_one_fram(self):
#         ret, frame = self.cap.read()
#         img = cv2.resize(frame, self.resize_shape)
#         img = img[..., ::-1]
#






cap = cv2.VideoCapture(0)
while (1):
    ret, frame = cap.read()
    k = cv2.waitKey(1)
    # if k == ord('p'):
    frame = deal_one_img(frame, "img")
    cv2.imshow("cap", frame)
    # cv2.imwrite("mmx.jpg",frame)
    cv2.waitKey(10)
    # else:
    #     cv2.imshow("cap", frame)
