#水増し部分
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# 1枚あたり20枚の画像を水増し
N_img = 10

# 入力画像の保存先パス
input_path = "C:\\Users\\saitowork\\anaconda3\\envs\\gazou2\\vector analysisf"
files = glob.glob(input_path + "\\*.JPG")
 
# 出力画像の保存先パス
output_path = "C:\\Users\\saitowork\\anaconda3\\envs\\gazou2\\vector analysisf"
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)
 
 
for i, file in enumerate(files):
 
    img = load_img(file)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
 
    # ImageDataGeneratorの生成
    datagen = ImageDataGenerator(
    zca_epsilon=1e-01,   # 白色化のイプシロン
    rotation_range=0.3, # ランダムに回転させる範囲
    width_shift_range=1.2, # ランダムに幅をシフトさせる範囲
    height_shift_range=1.3, # ランダムに高さをシフトさせる範囲
    brightness_range=None, # ランダムに明るさを変化させる範囲
    zoom_range=0.5,        # ランダムにズームさせる範囲
    horizontal_flip=True, # ランダムに水平方向に反転させる
    #vertical_flip=None, # ランダムに垂直方向に反転させる
    )
 
    # 1枚あたり20枚の画像を水増し生成
    dg = datagen.flow(x, batch_size=1, save_to_dir=output_path, save_prefix='img', save_format='JPG')
    for i in range(N_img):
        batch = dg.next()