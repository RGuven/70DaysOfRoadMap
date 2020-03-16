
# %92~%94 accuracy but our dataset is very poor. We have got only 50 images (25 Normal / 25 Covid) but I will improve this work.I want to feed this dataset.

from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2


#imagePath=r"dataset\covid\1-s2.0-S0140673620303706-fx1_lrg.jpg" # Covid => OUTPUT : [0] so This is correct.
imagePath=r"dataset\normal\IM-0033-0001-0001.jpeg" #Normal => Output: [1] so This is correct too.



model=load_model("CovidWeights.h5")

image = cv2.imread(imagePath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
image=image.reshape((-1,224,224,3))
image = np.array(image) / 255.0

y_pred = model.predict(image, batch_size=8)
predict = np.argmax(y_pred, axis=1)

print(predict)