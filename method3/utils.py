import numpy as np

def makeborder(small_image, large_image, fill= int):
    '''fill: the grayscale of border\n
    ret-> (tuple) expanded image of small image, exp_y, exp_x'''
    exp_y, exp_x = 0, 0
    if small_image.shape[0] < large_image.shape[0]:
        exp_y = large_image.shape[0]-small_image.shape[0]
    if small_image.shape[1] < large_image.shape[1]:
        exp_x = large_image.shape[1]-small_image.shape[1]

    datatype = small_image.dtype
    expanded = np.vstack((fill*np.ones((exp_y, small_image.shape[1]), dtype= datatype), small_image, fill*np.ones((exp_y, small_image.shape[1]), dtype= datatype)))
    expanded = np.hstack((fill*np.ones((expanded.shape[0], exp_x), dtype= datatype), expanded, fill*np.ones((expanded.shape[0], exp_x), dtype= datatype)))
    return expanded, exp_y, exp_x

def cropborder(image, exp_y, exp_x):
    return(image[exp_y:-exp_y, exp_x:-exp_x])