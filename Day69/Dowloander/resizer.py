"""

Resimler 200 Kb'den fazla olmamalı.
720x1280 den büyük olmamalı.

Ben sabit 640x480 yapıyorum.

"""

import numpy as np
import cv2
import os

dir_path = os.getcwd()
counter=0
for filename in os.listdir(dir_path):
    
    print(f"You are {counter}. image")
    counter+=1

    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".JPEG") or filename.endswith(".JPG") :
        image = cv2.imread(filename)
        #resized = cv2.resize(image,None,fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

        if image.shape[0]<480 or image.shape[1]<640:
            print("Image not changed because small")
            cv2.imwrite(str(counter)+".jpg",image)
            os.remove(filename)
            continue
        else:
            resized = cv2.resize(image,(640,480),interpolation = cv2.INTER_AREA)
            cv2.imwrite(str(counter)+".jpg",resized)
            os.remove(filename)

    elif not filename.endswith(".jpg") or filename.endswith(".jpeg"):
        print("This image not changed -->",filename)
        if filename.endswith(".png"):
            print("!!!   image PNG --->JPG changing   !!!")

            image_png = cv2.imread(filename)

            # Save .jpg image
            resized = cv2.resize(image_png,(640,480),interpolation = cv2.INTER_AREA)
            cv2.imwrite(str(counter)+".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 100]) #100 kalite ile scale ile ilgili default 95 [0..100]
            print(" \nIMAGE CHANGED\n ")

            #after delete png file
            os.remove(filename)

    
    else:
        print("IMAGEs NOT CHANGED !!!")
