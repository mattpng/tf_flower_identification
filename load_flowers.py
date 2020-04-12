import tensorflow as tf
print(tf.__version__) # should be 2.0
# assert tf.__version__.startswith('2')

import IPython.display as display
# from PIL import Image # not working
import numpy as np
import matplotlib.pyplot as plt
import os
import ssl

# Method 1: TensorFlow Core Load images tutorial
# https://www.tensorflow.org/tutorials/load_data/images
# AUTOTUNE = tf.data.experimental.AUTOTUNE # not working

# Method 2: TFLite Model Maker (for TensorFlow v2.x)
#https://www.tensorflow.org/tutorials/load_data/images
# from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
# from tensorflow_examples.lite.model_maker.core.task import image_classifier
# from tensorflow_examples.lite.model_maker.core.task.model_spec import mobilenet_v2_spec
# from tensorflow_examples.lite.model_maker.core.task.model_spec import ImageModelSpec


# Step 1: Retrieve images 
import pathlib
data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
data_dir = pathlib.Path(data_dir)

# Check that load is successful
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Image count ", image_count)

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
print(CLASS_NAMES)

# roses = list(data_dir.glob('roses/*'))
# for image_path in roses[:3]:
#     display.display(Image.open(str(image_path)))  # can't import Image













# # Step 2: Load using tf.data. Doesn't work because AUTOTUNE 
# list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

# def get_label(file_path):
#   # convert the path to a list of path components
#   parts = tf.strings.split(file_path, os.path.sep)
#   # The second to last is the class-directory
#   return parts[-2] == CLASS_NAMES

# def decode_img(img):
#   # convert the compressed string to a 3D uint8 tensor
#   img = tf.image.decode_jpeg(img, channels=3)
#   # Use `convert_image_dtype` to convert to floats in the [0,1] range.
#   img = tf.image.convert_image_dtype(img, tf.float32)
#   # resize the image to the desired size.
#   return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

# def process_path(file_path):
#   label = get_label(file_path)
#   # load the raw data from the file as a string
#   img = tf.io.read_file(file_path)
#   img = decode_img(img)
#   return img, label

# # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
# labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE) # AUTOTUNE is broke :(

# for image, label in labeled_ds.take(1):
#   print("Image shape: ", image.numpy().shape)
#   print("Label: ", label.numpy())