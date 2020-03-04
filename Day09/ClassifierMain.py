import cv2
import numpy as np

#Setup classifier



cup=cv2.CascadeClassifier('cups.xml')
                                    
cap=cv2.VideoCapture(0)

counter=0
cup_counter=0
while True:
    ret, img=cap.read()

    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  

    
    cup_detection=cup.detectMultiScale(gray,3,5)
    
 

    for (x,y,w,h) in cup_detection:
        
        cup_counter=cup_counter+1
        print("Cup Detecteddddddddddd...................")
        print(cup_counter)
   
	cv2.imshow('img', img)
    if (cv2.waitKey(30) & 0xff)==27:
        break

cap.release()
cv2.destroyAllWindows()
