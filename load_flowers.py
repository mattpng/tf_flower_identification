import tensorflow as tf
# AUTOTUNE = tf.data.experimental.AUTOTUNE # not working

import IPython.display as display
# from PIL import Image # not working
import numpy as np
import matplotlib.pyplot as plt
import os
import ssl

# retrieve images 
import pathlib
data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
data_dir = pathlib.Path(data_dir)

# test
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
# CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
# print(CLASS_NAMES)

# roses = list(data_dir.glob('roses/*'))
# for image_path in roses[:3]:
#     display.display(Image.open(str(image_path)))