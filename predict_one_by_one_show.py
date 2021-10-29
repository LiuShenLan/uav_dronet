import os,sys
import utils
import cv2
import numpy as np
from common_flags import FLAGS

import math
import tensorflow_probability as tfp
tfd = tfp.distributions
from keras import backend as K
TEST_PHASE=0

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

def central_image_crop(img, crop_height,crop_width):
    """
    Crops the input PILLOW image centered in width and starting from the bottom
    in height.
    Arguments:
        crop_width: Width of the crop
        crop_height: Height of the crop
    Returns:
        Cropped image
    """
    half_the_width = int(img.shape[1] / 2)

    img = img[(img.shape[0] - crop_height): img.shape[0],
          int(half_the_width - (crop_width / 2)): int(half_the_width + (crop_width / 2))]
    if FLAGS.img_mode == 'grayscale':
        img = img.reshape((img.shape[0], img.shape[1], 1))
    return img

def sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    without_max = np.exp(x) / np.sum(np.exp(x))
    x_max = np.max(x, axis=axis, keepdims=True)
    with_max = np.exp(x - x_max) / np.sum(np.exp(x - x_max))
    print(without_max)
    print(with_max)
    return with_max

def gaussian(sigs, mus, pis, x):
    gmm = 0
    for sigma, u, pi in zip(sigs, mus, pis):
        y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
        # print(sigma,u,pi,x,y)
        gmm = gmm + y * pi
    return gmm

def main():
    # 设置模型加载路径
    json_model_path = os.path.join(sys.path[0], 'model/model_struct.json')
    weights_path = os.path.join(sys.path[0], 'model/download/model_weights_299.h5')
    dataset_dir = './dataset/drone-data-train/'

    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height
    target_size = (img_height, img_width)

    # Cropped image dimensions
    crop_img_width, crop_img_height = FLAGS.crop_img_width, FLAGS.crop_img_height
    crop_size = (crop_img_height, crop_img_width)

    # 设置 keras utils
    K.set_learning_phase(TEST_PHASE)
    # 加载 json 并创建模型
    model = utils.jsonToModel(json_model_path)
    # 加载权重
    model.load_weights(weights_path)

    model.compile(loss='mse', optimizer='adam')
    # model.compile(loss='mse', optimizer='sgd')

    # 显示窗口设置
    cv2.namedWindow("keras Img Predict", 0)
    cv2.resizeWindow("keras Img Predict", 960, 540)

    for foldername in os.listdir(dataset_dir):
        if os.path.isfile(foldername):
            continue

        print(foldername)
        pics_path = os.path.join(dataset_dir, foldername, 'images')

        # Input image dimensions
        img_width, img_height = FLAGS.img_width, FLAGS.img_height
        target_size = (img_height,img_width)

        # Cropped image dimensions
        crop_img_width, crop_img_height = FLAGS.crop_img_width, FLAGS.crop_img_height
        crop_size = (crop_img_height,crop_img_width)

        pic_list = os.listdir(pics_path)
        pic_list.sort()
        for count, pic in enumerate(pic_list):
            # select pic

            # for file in pic_list:
            #     print("{0}, {1}".format(count, file))
            #     count = count + 1
            # pic_index = input("Input the number of the pic:")
            #pic = pic_list[int(pic_index)]
            print(pic)
            img_origi = cv2.imread(os.path.join(pics_path, pic), cv2.IMREAD_COLOR)

            # 图像预处理
            if FLAGS.img_mode == 'grayscale':
                img = cv2.cvtColor(img_origi, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (target_size[1], target_size[0]))
            else:
                img = cv2.resize(img_origi, (target_size[1], target_size[0]))

            img = central_image_crop(img, crop_size[0], crop_size[1])
            if FLAGS.img_mode == 'grayscale':
                img = img.reshape((img.shape[0], img.shape[1], 1))

            cv_image = np.asarray(img, dtype=np.float32) * np.float32(1.0 / 255.0)

            # 模型预测
            outs = model.predict_on_batch(cv_image[None])
            parameter, translation = outs[0][0], outs[1][0]
            # print("steer = {}, translation = {}".format(parameter,translation))

            # 预测数据处理
            y_pred = np.reshape(parameter, [-1, 6])
            out_mu, out_pi = np.split(y_pred, 2, axis=1)
            pi = sum_exp(out_pi, 1)
            pi = np.split(pi, 3, axis=1)
            # component_splits = [1, 1, 1]
            mus = np.split(out_mu, 3, axis=1)

            out_sigma = np.array([[0.1, 0.1, 0.1]], dtype='float32')
            sigs = np.split(out_sigma, 3, axis=1)

            x = np.linspace(-1, 1, 100)
            y = np.array([])
            for x_ in x:
                y = np.append(y, gaussian(sigs, mus, pi, x_))

            possible_direct = []

            start = 0
            continue_flag = 0
            sum_y = 0
            sum_x = 0
            for x_, y_ in zip(x, y):
                # if y_ > 1.:
                if y_ > 0.6:
                    if continue_flag == 0:
                        continue_flag = 1
                        start = x_
                    sum_y = sum_y + y_
                    sum_x = sum_x + 1
                    y_ = (img_origi.shape[0] - y_ * 200 - 80).astype(np.int32)
                    x_ = ((x_ + 1) / 2 * img_origi.shape[1]).astype(np.int32)
                    x_ = img_origi.shape[1] - x_
                    # cv2.circle(img_origi, (x_, y_), 3, (0, 255, 0), 4)
                    cv2.circle(img_origi, (x_, int(y_ / 2) + 150), 3, (0, 255, 0), 4)
                else:
                    if continue_flag == 1:
                        continue_flag = 0
                        possible_direct.append((x_ + start) / 2)
                        sum_y = 0
                        sum_x = 0
                    y_ = (img_origi.shape[0] - y_ * 200 - 80).astype(np.int32)
                    x_ = ((x_ + 1) / 2 * img_origi.shape[1]).astype(np.int32)
                    x_ = img_origi.shape[1] - x_
                    # cv2.circle(img_origi, (x_, y_), 1, (255, 0, 255), 4)
                    cv2.circle(img_origi, (x_, int(y_ / 2) + 150), 1, (255, 0, 255), 4)
            count = 0

            for possible_direct_ in possible_direct:
                # print(possible_direct_)
                cv2.line(img_origi, (int(img_origi.shape[1] / 2), img_origi.shape[0] - 50),
                         (int(img_origi.shape[1] / 2 - math.tan(possible_direct_ * 3.14 / 2) * 100),
                          img_origi.shape[0] - 150),
                         (0, 255, 0), 3)
                count = count + 1

            # cv2.line(img_origi, (int(img_origi.shape[1]/2),img_origi.shape[0]), (int(img_origi.shape[1]/2),50), (0,255,0), 1)
            cv2.line(img_origi, (int(img_origi.shape[1] / 2), img_origi.shape[0] - 50),
                     (int((translation + 1) / 2 * img_origi.shape[1]), img_origi.shape[0] - 50), (255, 255, 0), 8)
            cv2.imshow("keras Img Predict", img_origi)
            cv2.imshow('crop', img)
            cv2.waitKey(0)

if __name__ == '__main__':
    main()