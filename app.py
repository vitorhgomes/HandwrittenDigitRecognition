from pywebio.input import *
from pywebio.output import *

import cv2
import numpy as np
from tensorflow import keras


def transform_image(img):
    img = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28,28))
    img = img.astype('float32')
    img /= 255
    img = img.reshape([1, 28, 28, 1])
    return img

model = keras.models.load_model('model.h5')

while True:
    clear()
    put_markdown("### Handwritten Digit Recognition")
    img = file_upload("Select the picture:", accept="image/*", multiple=False)
    the_image = img['content']

    put_image(the_image)

    the_image_array = transform_image(the_image)
    y = model.predict(the_image_array)[0]
    # put_text(y)
    
    put_text(f'The digit is {y.argmax()}').style('font-size: 200%')
    correct = checkbox(options=['Mark this box if the digit was guessed correctly.'])