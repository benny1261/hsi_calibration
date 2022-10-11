import cv2
import numpy as np
import glob
import os
import re

def shift(hsi, wl, shift, alpha = 125):
    '''shift -> allows a tuple (delta_y, delta_x), defined as (0,0) of wl relative to hsi's aligned\n
    alpha -> transparency
    @ret: a png format image'''
    # following values are based on number of pixels instead of pixel index
    x, y = shift
    dx, dy = -x, -y                     # change x, y as pixels that needs to be croped
    dx2, dy2 = wl.shape[1]-hsi.shape[1]-dx, wl.shape[0]-hsi.shape[0]-dy
    patch_up, patch_left, patch_down, patch_right = 0,0,0,0

    # slicing using index =====================================
    if dy > 0:
        wl = wl[dy:, :]
    else: patch_up = abs(dy)            # zero maintains zero
    if dx > 0:
        wl = wl[:, dx:]
    else: patch_left = abs(dx)
    if dy2 > 0:
        wl = wl[:-dy2, :]
    else: patch_down = abs(dy2)
    if dx2 > 0:
        wl = wl[:, :-dx2]
    else: patch_right = abs(dx2)

    # adding alpha layer=======================================
    wl = np.dstack((wl, alpha*np.ones((wl.shape[0], wl.shape[1]))))         # notice that opencv and numpy arrays are transpose relations

    # patching ================================================
    up_arr = np.dstack((np.zeros((patch_up, wl.shape[1]), dtype=np.uint8), np.zeros((patch_up, wl.shape[1]), dtype=np.uint8), np.zeros((patch_up, wl.shape[1]), dtype=np.uint8),
    np.zeros((patch_up, wl.shape[1]), dtype=np.uint8)))
    down_arr = np.dstack((np.zeros((patch_down, wl.shape[1]), dtype=np.uint8), np.zeros((patch_down, wl.shape[1]), dtype=np.uint8), np.zeros((patch_down, wl.shape[1]), dtype=np.uint8),
    np.zeros((patch_down, wl.shape[1]), dtype=np.uint8)))
    left_arr = np.dstack((np.zeros((patch_up+patch_down+wl.shape[0], patch_left), dtype=np.uint8), np.zeros((patch_up+patch_down+wl.shape[0], patch_left), dtype=np.uint8),
    np.zeros((patch_up+patch_down+wl.shape[0], patch_left), dtype=np.uint8), np.zeros((patch_up+patch_down+wl.shape[0], patch_left), dtype=np.uint8)))
    right_arr = np.dstack((np.zeros((patch_up+patch_down+wl.shape[0], patch_right), dtype=np.uint8), np.zeros((patch_up+patch_down+wl.shape[0], patch_right), dtype=np.uint8),
    np.zeros((patch_up+patch_down+wl.shape[0], patch_right), dtype=np.uint8), np.zeros((patch_up+patch_down+wl.shape[0], patch_right), dtype=np.uint8)))

    wl = np.vstack((up_arr, wl, down_arr))
    wl = np.hstack((left_arr, wl, right_arr))

    print("shape of final wl:", wl.shape)
    return wl

if __name__ == '__main__':

    # PARAM ============================================================================================
    COMPRESSION_RATIO = 0.5678              #9/16 =0.5625
    SHIFT = (165, -105)                     # (x, y)
    ALPHA = 125
    CALIBRATION = True

    # READ =============================================================================================
    os.chdir(os.path.dirname(os.path.abspath(__file__))+'/data')
    print(os.getcwd())
    img_list = glob.glob('*.jpg')+ glob.glob('*.png')
    hsi_dict = {}
    wl_dict = {}

    # Process===========================================================================================
    if not CALIBRATION:
        for i in img_list:
            if 'hsi' in i:
                hsi_dict[i.split(".")[0]] = cv2.imread(i, cv2.IMREAD_COLOR)
            elif 'wl' in i:
                wl_dict[i.split(".")[0]] = cv2.imread(i, cv2.IMREAD_COLOR)

        if (not hsi_dict) or (not wl_dict):
            raise IOError(FileNotFoundError, "insufficient required image")
        elif len(hsi_dict) != len(wl_dict):
            raise IOError("number of different fluorescent images are not equal")
        else:
            for key, value in hsi_dict.items():
                wl_key = key.replace("hsi", "wl")
                wl_resize = cv2.resize(wl_dict[wl_key], (0,0), fx= COMPRESSION_RATIO, fy= COMPRESSION_RATIO, interpolation= cv2.INTER_AREA)
                final = shift(value, wl_resize, shift= SHIFT, alpha= ALPHA)
                if re.search('._hsi', key):
                    filename = key.replace("_hsi","")+"_cali"
                else:
                    filename = key.replace("hsi","")+"_cali"
                cv2.imwrite(filename+'.png', final)

    # Reference ========================================================================================
    if CALIBRATION:
        hsi = cv2.imread("calibration10x_hsi.png", cv2.IMREAD_COLOR)
        wl = cv2.imread("calibration10x_wl.jpg", cv2.IMREAD_COLOR)

        wl_resize = cv2.resize(wl, (0,0), fx= COMPRESSION_RATIO, fy= COMPRESSION_RATIO, interpolation= cv2.INTER_AREA)
        wl_inv = 255*np.ones_like(wl_resize)-wl_resize
        cv2.imwrite("wl_mod.jpg", wl_inv)
        final = shift(hsi, wl_resize, shift= SHIFT, alpha= ALPHA)
        cv2.imwrite("calibration_result.png", final)