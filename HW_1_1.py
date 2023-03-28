import pandas as pd
import tensorflow as tf
from keras import Model
from keras import layers
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import os

HW1_1_folder = "D://Deep Learning//HW1-1_data"
blotch = "D://Deep Learning//HW1-1_data//blotch"
normal = "D://Deep Learning//HW1-1_data//normal"
rot = "D://Deep Learning//HW1-1_data//rot"
scab = "D://Deep Learning//HW1-1_data//scab"

blotch_folder = os.listdir(blotch)
categories1 = []
for filename in blotch_folder:
    categories1.append('blotch')

normal_folder = os.listdir(normal)
categories2 = []
for filename in normal_folder:
    categories2.append('normal')

rot_folder = os.listdir(rot)
categories3 = []
for filename in rot_folder:
    categories3.append('rot')

scab_folder = os.listdir(scab)
categories4 = []
for filename in scab_folder:
    categories4.append('scab')

filenames = np.concatenate((blotch_folder, normal_folder, rot_folder, scab_folder), axis=None)
categories = np.concatenate((categories1, categories2, categories3, categories4), axis=None)

df = pd.DataFrame({
    'img_name': filenames,
    'category': categories
})
img_arr = []
for x in range(len(df)):
    img_arr.append([df.iloc[0], df.iloc[1]])

print(img_arr)
np.random.shuffle(img_arr)
