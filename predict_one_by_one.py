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
    # print(sys.path[0])
    # file_name(sys.path[0])
    json_model_path = os.path.join(sys.path[0], 'model/model_struct.json')
    weights_path = os.path.join(sys.path[0], 'model/train1/model_weights_299.h5')
    # 设置 keras utils
    K.set_learning_phase(TEST_PHASE)
    # 加载 json 并创建模型
    model = utils.jsonToModel(json_model_path)
    # 加载权重
    model.load_weights(weights_path)

    model.compile(loss='mse', optimizer='adam') # model.compile(loss='mse', optimizer='sgd')

    print("json_model_path: {}".format(json_model_path))
    print("Loaded model from {}".format(weights_path))

    # print("[INFO]")
    # model.summary()
    cv2.namedWindow("img", 0);
    cv2.resizeWindow("img", 960, 540);
    dataset_dir = './dataset/drone-data-test/'

    for foldername in os.listdir(dataset_dir):
        if os.path.isfile(foldername):
            continue

        if(foldername[0] != 't' and foldername[len(foldername)-1] != 'p'):
            print(foldername)
            pics_path = os.path.join(dataset_dir, foldername, 'images')

            dirct_label_exist = 1
            trans_label_exist = 1

            direct_label_path = os.path.join(dataset_dir, foldername, "direction_n_filted.txt")
            trans_label_path = os.path.join(dataset_dir, 'translation' + foldername, "translation.txt")
            try:
                direct_label = np.loadtxt(direct_label_path, usecols=0)
            except OSError as e:
                dirct_label_exist = 0
                print('No direction labels.')

            try:
                trans_label = np.loadtxt(trans_label_path, usecols=0)
            except OSError as e:
                trans_label_exist = 0
                print('No translation labels.')


            # Input image dimensions
            img_width, img_height = FLAGS.img_width, FLAGS.img_height
            target_size = (img_height,img_width)

            # Cropped image dimensions
            crop_img_width, crop_img_height = FLAGS.crop_img_width, FLAGS.crop_img_height
            crop_size = (crop_img_height,crop_img_width)


            correct_dirct_num = 0
            correct_trans_num = 0
            count = 0

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

                # run predict
                if FLAGS.img_mode == 'grayscale':
                    img = cv2.cvtColor(img_origi, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (target_size[1], target_size[0]))
                else:
                    img = cv2.resize(img_origi, (target_size[1], target_size[0]))

                img = central_image_crop(img, crop_size[0], crop_size[1])
                if FLAGS.img_mode == 'grayscale':
                    img = img.reshape((img.shape[0], img.shape[1], 1))

                cv_image = np.asarray(img, dtype=np.float32) * np.float32(1.0/255.0)

                # print(cv_image)
                outs = model.predict_on_batch(cv_image[None])
                # print(len(outs[0]))
                parameter, translation = outs[0][0], outs[1][0]
                print("steer = {}, translation = {}".format(parameter,translation))

                y_pred = np.reshape(parameter, [-1, 6])
                out_mu, out_pi = np.split(y_pred, 2, axis=1)
                # print(out_pi)
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
                for x_, y_ in zip(x, y):
                    # print(point)
                    if(y_ > 1.):
                        if(continue_flag == 0):
                            continue_flag = 1
                            start = x_
                        y_ = (img_origi.shape[0] - y_ * 200 - 80).astype(np.int32)
                        x_ = ((x_ + 1) / 2 * img_origi.shape[1]).astype(np.int32)
                        x_ = img_origi.shape[1] - x_
                        cv2.circle(img_origi, (x_, y_), 3, (0, 255, 0), 4)
                    else:
                        if(continue_flag == 1):
                            continue_flag = 0
                            possible_direct.append((x_ + start)/2)
                        y_ = (img_origi.shape[0] - y_ * 200 - 80).astype(np.int32)
                        x_ = ((x_ + 1) / 2 * img_origi.shape[1]).astype(np.int32)
                        x_ = img_origi.shape[1] - x_
                        cv2.circle(img_origi, (x_, y_), 1, (255, 0, 255), 4)

                # cat = tfd.Categorical(logits=out_pi)
                # coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                #         in zip(mus, sigs)]
                # mixture = tfd.Mixture(cat=cat, components=coll)
                # with tf.Session() as sess:
                #     xx = tf.expand_dims(tf.linspace(-1., 1., int(1e2)), 1)
                #     yy = mixture.prob(xx).eval()
                #     xx = tf.cast(((xx+1)/2*img_origi.shape[1]), dtype=tf.int32).eval()
                #     yy = tf.cast(img_origi.shape[0]-yy*200-80, dtype=tf.int32).eval()
                #     for point in zip(xx, yy):
                #         # print(point)
                #         cv2.circle(img_origi, point, 1, (0, 0, 255), 4)
                #     # plt.plot(x, mixture.prob(x).eval());
                #     # plt.savefig("abc.png")
                #

                if(dirct_label_exist):
                    steer = direct_label[count]
                    print('direction label: {}'.format(steer))
                    cv2.line(img_origi, (int(img_origi.shape[1]/2), img_origi.shape[0]-150), (int(img_origi.shape[1]/2 - math.tan(steer*3.14/2)*100), img_origi.shape[0] - 250), (255,0,0), 3)
                    steer_x = ((steer + 1) / 2 * img_origi.shape[1]).astype(np.int32)
                    steer_x = img_origi.shape[1] - steer_x
                    steer_y = gaussian(sigs, mus, pi, steer)
                    steer_y = (img_origi.shape[0] - steer_y * 200 - 80).astype(np.int32)

                    steer_x_left = ((steer+0.1 + 1) / 2 * img_origi.shape[1]).astype(np.int32)
                    steer_x_left = img_origi.shape[1] - steer_x_left
                    steer_y_left = gaussian(sigs, mus, pi, steer+0.1)
                    steer_y_left = (img_origi.shape[0] - steer_y_left * 200 - 80).astype(np.int32)

                    steer_x_right = ((steer - 0.1 + 1) / 2 * img_origi.shape[1]).astype(np.int32)
                    steer_x_right = img_origi.shape[1] - steer_x_right
                    steer_y_right = gaussian(sigs, mus, pi, steer - 0.1)
                    steer_y_right = (img_origi.shape[0] - steer_y_right * 200 - 80).astype(np.int32)
                    # print('x:{}, y:{}'.format(steer_x, steer_y))
                    if (steer_y < 2*img_origi.shape[0]/3):
                        cv2.circle(img_origi, (steer_x,steer_y), 6, (255, 0, 0), 6)
                        cv2.circle(img_origi, (steer_x_left, steer_y_left), 6, (255, 0, 0), 6)
                        cv2.circle(img_origi, (steer_x_right, steer_y_right), 6, (255, 0, 0), 6)

                        correct_dirct_num = correct_dirct_num+1
                    else:
                        cv2.circle(img_origi, (steer_x,steer_y), 3, (0, 0, 255), 6)
                        cv2.circle(img_origi, (steer_x_left,steer_y_left), 3, (0, 0, 255), 6)
                        cv2.circle(img_origi, (steer_x_right,steer_y_right), 3, (0, 0, 255), 6)

                    # cv2.line(img_origi, (int(steer), img_origi.shape[0] - 150),
                    #          (int(steer), 50), (255, 0, 0), 4)

                # pics in /translation*/images are not the same as those pics in direction dataset.
                # (if they are same, following code can be used)
                # if(trans_label_exist):
                #     trans = trans_label[count]
                #     if(math.fabs(translation - trans)<0.2):
                #         correct_trans_num = correct_trans_num+1
                #     print('translation label: {}'.format(trans))
                #     cv2.line(img_origi, (int(img_origi.shape[1] / 2), img_origi.shape[0] - 150),
                #              (int((trans + 1) / 2 * img_origi.shape[1]), img_origi.shape[0] - 150), (255, 0, 0), 8)
                cv2.line(img_origi, (int(img_origi.shape[1]/2),img_origi.shape[0]), (int(img_origi.shape[1]/2),50), (0,255,0), 1)
                cv2.line(img_origi, (int(img_origi.shape[1]/2),img_origi.shape[0]-150), (int((translation+1)/2*img_origi.shape[1]), img_origi.shape[0] - 150), (0,255,0), 8)
                for possible_direct_ in possible_direct:
                    print(possible_direct_)
                    cv2.line(img_origi, (int(img_origi.shape[1] / 2), img_origi.shape[0] - 150),
                         (int(img_origi.shape[1] / 2 - math.tan(possible_direct_ * 3.14 / 2) * 100), img_origi.shape[0] - 250),
                         (0, 255, 0), 3)

                cv2.imshow("img", img_origi)
                cv2.imshow('crop',img)
                cv2.waitKey(0)

            print('==================================')
            print('direct_accuracy = {}'.format(correct_dirct_num/len(pic_list)))
            print('trans_accuracy = {}'.format(correct_trans_num/len(pic_list)))


if __name__ == '__main__':
    main()