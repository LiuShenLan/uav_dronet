import os
import cv2
import tensorflow as tf
import img_utils
import random
pics_path = "/home/rikka/uav-project/drone-data-train/D/images/"
pic_list = os.listdir(pics_path)
pic_list.sort()
for img_name in pic_list:
    img_show = img_utils.load_img(pics_path + img_name)
    oriten = random.randint(0,2)
    trans = random.randint(0, 2)

    p1 = (int(img_show.shape[1] / 2), 280)
    p2 = (int(img_show.shape[1] / 2 + (oriten - 1) * 80), 200 + 32 * (abs(oriten-1)-1))
    p3 = (int(img_show.shape[1] / 2) + (trans - 1) * 80, 280)
    cv2.line(img_show, p1, p2, (0, 255, 0), 3)
    cv2.line(img_show, p1, p3, (0, 0, 255), 3)
    cv2.imshow("win",img_show/255)
    cv2.waitKey(0)