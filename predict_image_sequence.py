import os,sys
import utils
import cv2
import numpy as np
import math

from keras import backend as K
TEST_PHASE=0

def central_image_crop(img, crop_width, crop_heigth):
    """
    Crops the input PILLOW image centered in width and starting from the bottom
    in height.
    Arguments:
        crop_width: Width of the crop
        crop_heigth: Height of the crop
    Returns:
        Cropped image
    """
    half_the_width = int(img.shape[1] / 2)

    img = img[(img.shape[0] - crop_heigth): img.shape[0],
          int(half_the_width - (crop_width / 2)): int(half_the_width + (crop_width / 2))]
    img = img.reshape(img.shape[0], img.shape[1], 1)
    return img

def main():
    # print(sys.path[0])
    # file_name(sys.path[0])
    json_model_path = sys.path[0] + '/model/mytest_7_net_changed/model_struct.json'
    weights_path = sys.path[0] + '/model/mytest_7_net_changed/model_weights_249.h5'
    pics_path = sys.path[0] + '/pics'
    target_size = (320,240)
    crop_size = (200,200)


    # Set keras utils
    K.set_learning_phase(TEST_PHASE)
    # Load json and create model
    model = utils.jsonToModel(json_model_path)
    # Load weights
    model.load_weights(weights_path)
    #model.compile(loss='mse', optimizer='sgd')
    model.compile(loss='mse', optimizer='adam')

    print("json_model_path: {}".format(json_model_path))
    print("Loaded model from {}".format(weights_path))

    while True:
        # select pic
        count = 0
        pic_list = os.listdir(pics_path)
        pic_list.sort()
        for file in pic_list:
            print("{0}, {1}".format(count, file))
            count = count + 1
            img = cv2.imread(os.path.join(pics_path, file), cv2.IMREAD_COLOR)
            # run predict
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = central_image_crop(img, crop_size[0], crop_size[1])
            cv_image = np.asarray(img, dtype=np.float32) * np.float32(1.0/255.0)
            # print(cv_image)
            outs = model.predict_on_batch(cv_image[None])
            print(len(outs[0]))
            steer, translation = outs[0][0], outs[1][0]
            print("steer = {}, translation = {}".format(steer,translation))
            cv2.line(img, (int(img.shape[0]/2),img.shape[1]), (int(img.shape[0]/2 - math.tan(steer*3.14/2)*30), img.shape[1] - 30), (0,0,255), 2)
            cv2.line(img, (int(img.shape[0]/2),img.shape[1]-10), (int((translation+1)/2*img.shape[0]), img.shape[1] - 10), (0,0,255), 2)
            cv2.imshow("pic", img)
            cv2.waitKey(300)

if __name__ == '__main__':
    main()
