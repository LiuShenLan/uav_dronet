# ——————————————— [1] 图片生成器的定义 ———————————————

from keras.preprocessing.image import ImageDataGenerator

# 指定参数
# rotation_range 旋转
# width_shift_range 左右平移
# height_shift_range 上下平移
# zoom_range 随机放大或缩小

img_generator = ImageDataGenerator(
    # rotation_range = 90,
     width_shift_range = 0.2,
    # height_shift_range = 0.2,
     zoom_range = [0.8, 1]
)
# ImageDataGenerator()中不设置参数就表示 不操作，后面生成batch就是直接在原数据集上抽取图片。

# ——————————————————————————————————————————————————

# ————————————————— [2] 输入数据的准备 —————————————————

from keras.preprocessing import image
import matplotlib.pyplot as plt

img_path1 = './dataset/fy/drone-data-test/GS22.1/images/frame-000000.jpg'
img_path2 = './dataset/fy/drone-data-test/GS22.1/images/frame-000002.jpg'
img1 = image.load_img(img_path1)  # plt格式的图片。
img2 = image.load_img(img_path2)  # plt格式的图片。

plt.imshow(img1)
plt.show()

# 将图片转为数组
image_list = []  # 模拟只有2张图片的数据集。
image_list.append(image.img_to_array(img1))  # .img_to_array(img1)为将图片转化成数组。
image_list.append(image.img_to_array(img2))  # .img_to_array(img2)为将图片转化成数组。

# 将数据集转为数组
import numpy as np

image_array = np.array(image_list)
print(image_array)
print(image_array.shape)

# [样本设置]
x_train = image_array  # 输入图片。
y_train = [1, 2]  # 标签

# [生成图片]： 其中，gen可以作为生成器，用model.fit_generate(generate,)中来训练。
gen = img_generator.flow(x_train, y_train, batch_size=2)  # x_train —— 要求类型:numpy.array; 要求形状: (image_num, 长, 宽, 通道)
# y_train —— 要求类型:numpy.array; 要求形状: (image_num)
# 注: (1) 每个batch中生成的图片是 从数据集的所有图片中,随机抽取一张并进行图片尺寸大小啥的变换后放入batch中, 这样抽取batch_size张图片后就形成一个batch。
#    (2) 对图片进行旋转尺寸大小变换后的图片,图片大小[不会]改变。

# 看看每个batch中生成的图片都是咋样的
plt.figure()
for i in range(10):
    x_batch, y_batch = next(gen)  # 每次 next() 一下, 返回一个batch: x_batch, y_batch。
    # x_batch —— 形状: (batch_size, 长, 宽, 通道);
    # y_batch —— 形状: (batch_size)
    print('\n')
    print('x_batch', x_batch)
    print('\n')
    print('x_batch类型', x_batch.__class__)
    print('\n')
    print('x_batch.shape:', x_batch.shape)
    print('\n')
    print('y_batch', y_batch)
    print('\n')

    plt.imshow(x_batch[0] / 256)  # batch中的第 1 张图片。
    plt.imshow(x_batch[1] / 256)  # batch中的第 2 张图片。
    plt.show()
