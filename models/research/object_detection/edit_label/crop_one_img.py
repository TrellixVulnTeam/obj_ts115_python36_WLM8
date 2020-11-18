import os

import cv2

src_img_dir = "/home/db/图片/刚棒入水实验/water_steal"
result_img_dir = "/home/db/图片/刚棒入水实验/reshape_img"
crop_size = (2250, 4000)

def crop_img(img):
    img_shape = img.shape
    img_h_start = int((img_shape[0] - crop_size[0]) / 2)
    img_w_start = int((img_shape[1] - crop_size[1]) / 2)
    img_result = img[img_h_start:img_h_start + crop_size[0], img_w_start:img_w_start + crop_size[1], :]
    # cv2.imshow("sdfa",img_result)
    print(img_result.shape)
    # cv2.waitKey(0)
    return img_result


if __name__ == "__main__":
    for root, dir, files in os.walk(src_img_dir):
        print(root)
        print(dir)
        for file in files:
            src_img_path = os.path.join(root, file)
            result_img_path = os.path.join(result_img_dir, file)
            src_img = cv2.imread(src_img_path)

            result_img = crop_img(src_img)
            cv2.imwrite(result_img_path,result_img)
            print(src_img_path)
            print(result_img_path)
