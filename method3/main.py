import numpy as np
import cv2
import os
import utils

CLIP_LIMIT = 4
TILEGRIDSIZE = 8
blur_kernal = (3, 3)
dilate_kernal = (3, 3)

os.chdir('data/')
image = cv2.imread("hsi_cali.png", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("wl_cali.png", cv2.IMREAD_GRAYSCALE)

# Preprocessing============================================================================
clahe = cv2.createCLAHE(clipLimit= CLIP_LIMIT, tileGridSize= (TILEGRIDSIZE, TILEGRIDSIZE))

image_cla = clahe.apply(image)
image_blr = cv2.GaussianBlur(image_cla, blur_kernal, 0)
ret, image_th = cv2.threshold(image_blr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite("img.jpg", image_th)

template = cv2.resize(template, (2508, 1411), interpolation= cv2.INTER_CUBIC)   # resize
# template = clahe.apply(template)
template_blr = cv2.GaussianBlur(template, blur_kernal, 0)
ret, temp_th = cv2.threshold(template_blr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
temp_th = cv2.dilate(temp_th, dilate_kernal, iterations= 1)                 # white dilate so black line erodes
cv2.imwrite("tem.jpg", temp_th)

# Expanding ref image======================================================================
'''
template must be smaller or equal than image in size!!!
hsi: 1536*2048
wl_resize: 1411*2508
'''
th_ex = utils.makeborder(image_th, temp_th, 255)
result = cv2.matchTemplate(th_ex, temp_th, cv2.TM_CCOEFF_NORMED)
print('th_ex:',th_ex.shape, 'result shape:', result.shape)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)        #  (X, Y) = Loc, can use mask??
print("Maxloc:", maxLoc)

# Appling calculated shift on original image===============================================
final, delta = utils.overlay_coord(image, template, maxLoc)

print("shift=", delta)                              # (x, y)
cv2.imwrite("output.jpg", final)
cv2.waitKey(0)