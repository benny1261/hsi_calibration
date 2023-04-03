import numpy as np

def makeborder(small_image, large_image, fill= int):
    '''fill: the grayscale of border\n
    ret-> expanded image of small image'''
    exp_y, exp_x = 0, 0
    if small_image.shape[0] < large_image.shape[0]:
        exp_y = large_image.shape[0]-small_image.shape[0]
    if small_image.shape[1] < large_image.shape[1]:
        exp_x = large_image.shape[1]-small_image.shape[1]

    datatype = small_image.dtype
    expanded = np.vstack((fill*np.ones((exp_y, small_image.shape[1]), dtype= datatype), small_image, fill*np.ones((exp_y, small_image.shape[1]), dtype= datatype)))
    expanded = np.hstack((fill*np.ones((expanded.shape[0], exp_x), dtype= datatype), expanded, fill*np.ones((expanded.shape[0], exp_x), dtype= datatype)))
    return expanded

def cropborder(image, exp_y, exp_x):
    return(image[exp_y:-exp_y, exp_x:-exp_x])

def overlay_coord(backgd_image, top_image, coord:tuple) -> tuple:
    '''coord: coordinate of (0, 0) of top_image on 'expanded' backgd_image (x, y)\n
    ret-> (tuple) overlay image, (delta_x, delta_y)'''

    cx, cy = coord
    delta_y, delta_x = cy- abs(backgd_image.shape[0]-top_image.shape[0]), cx- abs(backgd_image.shape[1]-top_image.shape[1])     # negative -> crop top image
    delta_y2, delta_x2 = backgd_image.shape[0]- delta_y - top_image.shape[0], backgd_image.shape[1]- delta_x - top_image.shape[1]

    rangy = Ranging(delta_y, delta_y2)
    rangx = Ranging(delta_x, delta_x2)
    backgd_image[rangy.bg1:rangy.bg2, rangx.bg1:rangx.bg2] = top_image[rangy.ov1:rangy.ov2, rangx.ov1:rangx.ov2]

    return backgd_image, (delta_x, delta_y)

class Ranging:
    def __init__(self, del1: int, del2: int) -> None:
        self.bg1 = None
        self.bg2 = None
        self.ov1 = None
        self.ov2 = None

        if del1 >= 0:
            self.bg1 = del1
        else:
            self.ov1 = -del1

        if del2 >= 0:
            self.bg2 = -del2
        else:
            self.ov2 = del2