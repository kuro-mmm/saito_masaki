#1 ライブラリのインポート等

from keras.models import load_model
import os
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob

###
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
 # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
 try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*1.5)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
 except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
###

#2 各種設定

recognise_image = "anaconda3\\envs\\gazou2\\Nora_Princess_and_Stray_cat\\img_0_1186.JPG"
                              # ここを変更
                           # 画像認識したい画像ファイル名。（実行前に認識したい画像ファイルを1つアップロードしてください）

#image_width = 100   # ここを変更
                   # 利用する学習済みモデルの横の幅のピクセル数と同じにする
#image_height = 800  # ここを変更
                   # 利用する学習済みモデルの縦の高さのピクセル数と同じにする
image_width = 90
image_height = 700

color_setting = 1  # ここを変更。利用する学習済みモデルのカラー形式と同じにする
                   # 「1」はモノクロ・グレースケール。「3」はカラー。


#3 各種読み込み

model = load_model('model2.h5')  # ここを変更
                                # 読み込む学習済みモデルを入れます

# モノクロ・グレー形式の学習済みモデルを読み込む例：color_setting = 1 の学習済みモデルを使う場合  
#model = load_model('keras_cnn_japanese_handwritten_gray14*14_model.h5')  

# カラー形式の学習済みモデルを読み込む例：color_setting = 3 の学習済みモデルを使う場合  
#model = load_model('keras_cnn_japanese_handwritten_color14*14_model.h5') 


#4 画像の表示・各種設定等

if color_setting == 1:
  img = cv2.imread(recognise_image, 0)   
elif color_setting == 3:
  img = cv2.imread(recognise_image, 1)
img = cv2.resize(img, (image_width, image_height))
plt.imshow(img)
if color_setting == 1:
  plt.gray()  
  plt.show()
elif color_setting == 3:
  plt.show()


img = img.reshape(image_width, image_height, color_setting).astype('float32')/255 


#5 予測と結果の表示等

prediction = model.predict(np.array([img]))
result = prediction[0]

folder = glob.glob(".\gazou\*")

print(result)
#(result)
for i, accuracy in enumerate(result):
  print('画像認識AIは「', os.path.basename(folder[i]), '」の確率を', int(accuracy * 100), '% と予測しました。')

print('-------------------------------------------------------')
print('予測結果は、「', os.path.basename(folder[result.argmax()]),'」です。')