import cv2

gray_img = cv2.imread('captain.png', cv2.IMREAD_GRAYSCALE)  # or put 0 
cv2.imshow('Grayscale', gray_img)
cv2.imwrite('output.png', gray_img)


cv2.waitKey()