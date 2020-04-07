import cv2

img = cv2.imread('captain.png')

cv2.imwrite('output_jpg_format.jpg', img)  # if ypur actual image format is jpg you can put after img, [cv2.IMWRITE_PNG_COMPRESSION] ..