# HSI Calibration

a program to calibrate pixel difference and shift between white light image and hyperspectral image

## current perameters

hsi:0.05mm = 90pixel
-> 1pixel= 0.555um (2048*1536)

wl:0.05mm = 160pixel
-> 1pixel = 0.3125um (4088*3072)

size>>>wl*(90/160) = 2300*1728

shift>>>center2center(-127,-96),x+294

cv2.resize: cubic for inlarge, area for shrink
