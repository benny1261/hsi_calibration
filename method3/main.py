import numpy as np
import cv2
import os
import utils

CLIP_LIMIT = 4
TILEGRIDSIZE = 8
blur_kernal = (3, 3)
dilate_kernal = (3, 3)

os.chdir('data/')
image = cv2.imread("calibration10x_hsi.png", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("wl_mod.jpg", cv2.IMREAD_GRAYSCALE)

# Preprocessing============================================================================
clahe = cv2.createCLAHE(clipLimit= CLIP_LIMIT, tileGridSize= (TILEGRIDSIZE, TILEGRIDSIZE))

image_cla = clahe.apply(image)
image_blr = cv2.GaussianBlur(image_cla, blur_kernal, 0)
ret, image_th = cv2.threshold(image_blr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite("img.jpg", image_th)

# template = clahe.apply(template)
template_blr = cv2.GaussianBlur(template, blur_kernal, 0)
ret, temp_th = cv2.threshold(template_blr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
temp_th = cv2.dilate(temp_th, dilate_kernal, iterations= 1)                 # white dilate so black line erodes
cv2.imwrite("tem.jpg", temp_th)

# Expanding ref image======================================================================
'''
template must be smaller in size than image!!!
hsi: 1536*2048
wl_resize: 1728*2300
'''
th_ex, expy, expx = utils.makeborder(image_th, temp_th, 255)
result = cv2.matchTemplate(th_ex, temp_th, cv2.TM_CCOEFF_NORMED)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

(startX, startY) = maxLoc
endY = startY + template.shape[0]
endX = startX + template.shape[1]
delta = (startX-expx, startY-expy)

# Appling calculated shift on original image===============================================
image_ex, _, _ = utils.makeborder(image, template, 255)
image_ex[startY:endY, startX:endX] = template
final = utils.cropborder(image_ex, expy, expx)

print("shift=", delta)                              # (x, y)
cv2.imwrite("output.jpg", final)
cv2.waitKey(0)