# HSI Calibration

a program to calibrate pixel difference and shift between white light image and hyperspectral image

## new parameters

*hsi: 0.2mm = 362pixel(2048x1536)*\
*wl: 0.2mm = 278pixel(1920x1080)\
*size: wl*(362/278) = 2500.14x1406.33 -> 2508x1411*\
*shift: delta_x. delta_y= (-178, 126)*\
*cv2.resize: cubic for inlarge, area for shrink*

**move range:**\
one block approx= 90*90 pixels\
no move more than half block = 45

****
