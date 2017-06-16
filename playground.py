from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.models import load_model
from PIL import Image as IMG
import numpy as np
import coremltools
import os

#
# pip packages:
# coremltools (0.3.0)
# funcsigs (1.0.2)
# h5py (2.7.0)
# Keras (1.2.2)
# mock (2.0.0)
# numpy (1.13.0)
# olefile (0.44)
# pbr (3.0.1)
# Pillow (4.1.1)
# pip (9.0.1)
# protobuf (3.3.0)
# PyYAML (3.12)
# scipy (0.19.0)
# setuptools (36.0.1)
# six (1.10.0)
# tensorflow (1.1.0)
# Theano (0.9.0)
# Werkzeug (0.12.2)
# wheel (0.29.0)
#

def printResults(all_results, num=3, key='probabilities'):
    probs=all_results[key]
    i = 0
    for key, value in sorted(probs.iteritems(), reverse=True, key=lambda (k,v): (v,k)):
        if i > num-1:
            return
        print "%s: %s" % (key, value)
        i += 1

elephant_pic = 'elephant.jpg'
peacock_pic = 'peacock.jpg'

model = InceptionV3(weights='imagenet', include_top=True)

elephant_img = IMG.open(elephant_pic)
peacock_img = IMG.open(peacock_pic)
elephant = image.img_to_array(elephant_img)
peacock = image.img_to_array(peacock_img)
elephant = np.expand_dims(elephant, axis=0)
peacock = np.expand_dims(peacock, axis=0)
elephant = preprocess_input(elephant)
peacock = preprocess_input(peacock)

elephant_preds = model.predict(elephant)
peacock_preds = model.predict(peacock)

print("KERAS")
print('Elephant Probabilities:\n', decode_predictions(elephant_preds, top=3))
print('Peacock Probabilities:\n', decode_predictions(peacock_preds, top=3))

scale = 2./255

coreml_model = coremltools.converters.keras.convert(model,
                                                    input_names=['image'],
                                                    output_names=['probabilities'],
                                                    image_input_names='image',
                                                    class_labels='classes.txt',
                                                    predicted_feature_name='class',
                                                    # is_bgr=True,
                                                    image_scale=scale,
                                                    red_bias=-1,
                                                    green_bias=-1,
                                                    blue_bias=-1)


print("CoreML Converted")
print("Elephant Probabilities:")
printResults(coreml_model.predict({'image': elephant_img}))
print("Peacock Probabilities:")
printResults(coreml_model.predict({'image': peacock_img}))

# In order to test with Apple's InceptionV3, download it from here:
# https://developer.apple.com/machine-learning/
# and place it in the same folder as this app, then uncomment this code:

# coreml_model = coremltools.models.MLModel('Inceptionv3.mlmodel')
# print("#########IMPORTED SPEC#########")
#
# print(coreml_model.get_spec())

#
# print("Apple InceptionV3:")
# print("Elephant Probabilities:")
# printResults(coreml_model.predict({'image': elephant_img}), key='classLabelProbs')
# print("Peacock Probabilities:")
# printResults(coreml_model.predict({'image': peacock_img}), key='classLabelProbs')

