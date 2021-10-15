import os,sys
import numpy as np
from common_flags import FLAGS
TEST_PHASE=0
import cnn_models
import img_utils
import cv2
from keras.utils import plot_model
def main():

    weights_path = sys.path[0] + '/model/weights_002.h5'
    pics_path = "/home/rikka/uav-project/drone-data-train/D/images"

    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height
    # Cropped image dimensions
    crop_img_width, crop_img_height = FLAGS.crop_img_width_res18, FLAGS.crop_img_height_res18
    target_size = (img_height,img_width)
    crop_size = (crop_img_height,crop_img_width)

    model = cnn_models.s_Resnet_18(crop_img_width,crop_img_height,3,1)
    plot_model(model,"model.png")
    model.load_weights(weights_path,by_name=True)
    model.compile(loss='mse', optimizer='adam')
    pic_list = os.listdir(pics_path)
    pic_list.sort()
    try:
        for img_name in pic_list:
            current_name = pics_path + '/' + img_name
            img = img_utils.load_img(current_name,target_size = target_size,crop_size = crop_size)
            img = np.asarray(img, dtype=np.float32) * np.float32(1.0 / 255.0)
            outs = model.predict_on_batch(img[None])
            oriten, trans = outs[0][0], outs[1][0]
            oriten = np.argmax(oriten)
            trans = np.argmax(trans)
            img_show = img_utils.load_img(current_name)
            p1 = (int(img_show.shape[1] / 2), 280)
            p2 = (int(img_show.shape[1] / 2 + (oriten - 1) * 80), 200 + 32 * (abs(oriten - 1) - 1))
            p3 = (int(img_show.shape[1] / 2) + (trans - 1) * 80, 280)
            cv2.line(img_show, p1, p2, (0, 255, 0), 3)
            cv2.line(img_show, p1, p3, (0, 0, 255), 3)
            cv2.imshow("win", img_show / 255)
            cv2.waitKey(10)
    except KeyboardInterrupt:
        print("calling to end")
if __name__ == '__main__':
    main()