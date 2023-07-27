#Import Libraries
#import tensorflow
import tensorflow  
from tensorflow import keras
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import os
import imutils

#print(tf.__version__)
#print("Num GPUs Available: ", 
#       len(tf.config.experimental.list_physical_devices('GPU')))
# Checking the version for incompatibilities and GPU list devices 
# for a fast check on GPU drivers installation. 

model_filepath = 'trained_models\\health_clssfy_inceptionv3.h5'

model = tensorflow.keras.models.load_model(
    model_filepath,
    custom_objects=None,
    compile=False
)
# load the model we saved
"""
model = load_model('C:\\Users\\Kanwal\\Kanwal UB work\\UB_prototypes\\deep_learning_models\\specie_clss_vgg16.h5')
model.compile(
  loss='categorical_crossentropy',
  optimizer = tfa.optimizers.AdamW(0.1, learning_rate),
  #optimizer='adam',
  #optimizer = keras.optimizers.Adam(lr=0.0001),
  metrics=['accuracy']
)
"""
def finder(image_path):
    fish_result=0
    sliced_image = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    input_arr = tensorflow.keras.preprocessing.image.img_to_array(sliced_image)
    input_arr = np.array([input_arr]) 
    input_arr = input_arr.astype('float32') / 255.  # This is VERY important
    predictions = model.predict(input_arr)
    print("predictions",predictions)

    predicted_class = np.argmax(predictions, axis=-1)
    print(predicted_class)
    results=predicted_class
    print("RESSSSSS", results)

    if (results[0]==0):
        fish_result="healthy"
    elif (results[0]==1):
        fish_result="unhealthy"

    print("fish_result",fish_result)

    return fish_result