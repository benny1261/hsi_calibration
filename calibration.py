import cv2
import numpy as np
import glob
import os

# PARAM ============================================================================================
COMPRESSION_RATIO = 1.3056              # 362/278 = 1.30216
SHIFT = (-178, 126)                     # (x, y)
ALPHA = 0.5
CALIBRATION = False
MASK = False

def resize(img:np.ndarray, ratio:float = COMPRESSION_RATIO):
    if ratio >= 1:
        resized = cv2.resize(img, (0,0), fx= ratio, fy= ratio, interpolation= cv2.INTER_CUBIC)
    else:
        resized = cv2.resize(img, (0,0), fx= ratio, fy= ratio, interpolation= cv2.INTER_AREA)

    return resized

def shifted_mask(hsi, wl, shift:tuple):
    '''shift -> allows a tuple (delta_x, delta_y), defined as (0,0) of wl relative to hsi's aligned\n
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
    wl = np.dstack((wl, 255*np.ones((wl.shape[0], wl.shape[1]))))           # notice that opencv and numpy arrays are transpose relations

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

    return wl

def shifted_combine(hsi:np.ndarray, wl:np.ndarray, shift:tuple, alpha:int = ALPHA):
    '''shift -> allows a tuple (delta_x, delta_y), defined as (0,0) of wl relative to hsi's aligned\n
    alpha -> transparency
    @ret: a png format image'''
    # following values are based on number of pixels instead of pixel index

    dx, dy = shift
    top, bottom, left, right = dy, dy+wl.shape[0], dx, dx+wl.shape[1]

    slice_t, slice_b = max(-top, 0), min(wl.shape[0], hsi.shape[0]-dy)
    slice_l, slice_r = max(-left, 0), min(wl.shape[1], hsi.shape[1]-dx)
    # slicing using index =====================================
    wl_slice = wl[slice_t:slice_b, slice_l:slice_r]
    hsi_overlay = np.s_[max(0, top):min(hsi.shape[0], bottom), max(0, left): min(hsi.shape[1], right)]
    hsi[hsi_overlay] = cv2.addWeighted(hsi[hsi_overlay], alpha, wl_slice, 1-alpha, gamma= 1)

    return hsi

if __name__ == '__main__':
    # READ =============================================================================================
    # os.chdir(os.path.dirname(os.path.abspath(__file__))+'/data')
    # img_list = glob.glob('*.jpg')+ glob.glob('*.png')
    # hsi_dict = {}
    # wl_dict = {}

    # Process===========================================================================================
    if not CALIBRATION:
        # for i in img_list:
        #     if 'hsi' in i and i != "hsi_cali.png":
        #         hsi_dict[i.split(".")[0]] = cv2.imread(i, cv2.IMREAD_COLOR)
        #     elif 'wl' in i and i != "wl_cali.png":
        #         wl_dict[i.split(".")[0]] = cv2.imread(i, cv2.IMREAD_COLOR)

        # if (not hsi_dict) or (not wl_dict):
        #     raise IOError(FileNotFoundError, "insufficient required image")
        # elif len(hsi_dict) != len(wl_dict):
        #     raise IOError("number of different fluorescent images are not equal")
        # else:
        #     for key, value in hsi_dict.items():
        #         wl_key = key.replace("hsi", "wl")
        #         wl_resize = resize(wl_dict[wl_key])
        #         filename = key.replace("hsi","")
        #         if MASK:
        #             final = shifted_mask(value, wl_resize, shift= SHIFT)
        #             cv2.imwrite(filename+'_mask.png', final)                    
        #         final2 = shifted_combine(value, wl_resize, shift= SHIFT)
        #         cv2.imwrite(filename+'_combine.png', final2)

        # --------------------------------------------------------------
        rootdir = os.path.dirname(__file__)+r'/data'

        for root, dirs, files in os.walk(rootdir):
            for dir_name in dirs:
                current_folder = os.path.join(root, dir_name)
                print(current_folder)
                os.chdir(current_folder)
                wl_list = glob.glob('*.png')
                hsi_list = glob.glob('*.jpg')
                if not all((wl_list, hsi_list)):
                    continue

                wl_img = cv2.imread('001.png', cv2.IMREAD_COLOR)
                hsi_img = cv2.imread(hsi_list[0], cv2.IMREAD_COLOR)
                filename = hsi_list[0].replace('.jpg', '')

                wl_img = resize(wl_img)
                out = shifted_combine(hsi_img, wl_img, shift= SHIFT)
                cv2.imwrite(filename+'_combine.png', out)

    # Reference ========================================================================================
    if CALIBRATION:
        hsi = cv2.imread("hsi_cali.png", cv2.IMREAD_COLOR)
        wl = cv2.imread("wl_cali.png", cv2.IMREAD_COLOR)

        wl_resize = resize(wl)
        final = shifted_mask(hsi, wl_resize, shift= SHIFT)
        final2 = shifted_combine(hsi, wl_resize, shift= SHIFT)
        cv2.imwrite("masked.png", final)
        cv2.imwrite("combined.png", final2)
    