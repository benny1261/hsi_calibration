import numpy as np
import cv2

image = cv2.imread("calibration10x_hsi.png", cv2.IMREAD_COLOR)
template = cv2.imread("wl_resize.jpg", cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)\

result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

(startX, startY) = maxLoc
endX = startX + template.shape[1]
endY = startY + template.shape[0]

cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)
try:
    image[startY:endY, startX:endX] = template
except:
    image[startY:, startX:] = template[:image.shape[0]-startY, :image.shape[1]-startX]
cv2.imwrite("output.jpg", image)

cv2.waitKey(0)