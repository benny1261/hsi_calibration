# HSI Calibration

a program to calibrate pixel difference and shift between white light image and hyperspectral image

## current perameters

*hsi: 0.05mm = 90pixel
-> 1pixel= 0.555um (2048x1536)*
\
*wl: 0.05mm = 160pixel
-> 1pixel = 0.3125um (4088x3072)*
\
*size: wl(90/160) = 2300x1728*
\
*shift: center2center(-127,-96),x+294*
\
*cv2.resize: cubic for inlarge, area for shrink*
****

## references

method 1:
using opencv [link](https://magamig.github.io/posts/accurate-image-alignment-and-registration-using-opencv/)

method 2:
using opencv [link](https://www.geeksforgeeks.org/image-registration-using-opencv-python/)

method 3:
using cv2.matchTemplate [link](https://pyimagesearch.com/2021/03/22/opencv-template-matching-cv2-matchtemplate/)

using matlab [link](https://www.mathworks.com/help/images/registering-an-image-using-normalized-cross-correlation.html?fbclid=IwAR0oiogTrI5Q_FoF52qB9MrFuFWO-bqjlAac4yXtU9uV527DuyuTEskvlGk)

more about ORB algorithm [link](https://zhuanlan.zhihu.com/p/261966288)
