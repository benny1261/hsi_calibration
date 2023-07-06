import os
import numpy as np
from spectral.io import envi
import spectral
import glob
import re
# import math
import cv2
from skimage import morphology, measure
# from skimage.segmentation import slic
# from skimage import future

'''
requires:
xxx.hdr
xxx.raw
xxx.txt(position info)
'''

OUTPUT_CHANNEL = 50
BOX_LENGTH = 32             # better use power of 2
# HALF_LEN = int(math.ceil(BOX_LENGTH-1)/2)
HALF_LEN = int(BOX_LENGTH/2)

def slice3d(hsi, cy: int, cx: int)-> np.ndarray:
    '''slice hsi into 3d numpy array'''
    if any((cy<HALF_LEN, cy>hsi.shape[0]-HALF_LEN, cx<HALF_LEN, cx>hsi.shape[1]-HALF_LEN)):
        print('position out of range')
    else:
        return np.array(hsi[cy-HALF_LEN:cy+HALF_LEN,cx-HALF_LEN:cx+HALF_LEN,:])

def automask(grayimg)-> np.ndarray:
    _, mask = cv2.threshold(grayimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # mask = slic(grayimg, n_segments=2, start_label = 0)
    # mask = future.manual_polygon_segmentation(grayimg)
    # mask = cv2.normalize(mask, None,0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    mask = cv2.bitwise_not(mask)                                    # inverse
    mask = morphology.convex_hull_object(mask, connectivity= 2)     # fill
    mask = measure.label(mask, connectivity= 2)                     # label
    center_coor = int((grayimg.shape[0]-1)/2)
    center_label = mask[center_coor, center_coor]
    binarymask = np.where(mask == center_label, 255, 0).astype(np.uint8)
    return binarymask

if __name__ == '__main__':
    rootdir = os.path.dirname(__file__)+r'/data'
    spectral.settings.envi_support_nonlowercase_params = 'TRUE'

    for root, dirs, files in os.walk(rootdir):
        for dir_name in dirs:
            current_folder = os.path.join(root, dir_name)
            print(current_folder)
            os.chdir(current_folder)
            hdr_list = glob.glob('*.hdr')
            hsi_dict = {}
            for _ in hdr_list:
                name = _.split(".")[0]
                try:
                    hsi_dict[name] = envi.open(name+ ".hdr" , name+ ".raw")     # our hsi metadata stored in ENVI raster format
                    arr = hsi_dict[name].load()
                    print(name,"loaded")
                    print("shape=", arr.shape)
                except:
                    print('cannot read hsi file of', name)
                    continue
                try:
                    with open(name+'.txt', 'r') as file:
                        position = file.read()
                        file.close()
                except:
                    print('no txt file of', name)
                    continue
                if position is not None:
                    rule_y = re.compile('(?<=y)\d+', re.IGNORECASE)
                    cys = [int(match.group()) for match in rule_y.finditer(position)]           # remember to convert to integer
                    rule_x = re.compile('(?<=x)\d+', re.IGNORECASE)
                    cxs = [int(match.group()) for match in rule_x.finditer(position)]
                    # cy = int(re.compile('(?<=y)\d+', re.IGNORECASE).search(position).group())
                    # cx = int(re.compile('(?<=x)\d+', re.IGNORECASE).search(position).group())
                    for index, (cy, cx) in enumerate(zip(cys, cxs)):
                        print(f'position = ({cy},{cx})')
                        nparr = slice3d(arr, cy, cx)
                        img_gray = cv2.normalize(nparr[:,:,OUTPUT_CHANNEL], None,0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
                        mask = automask(img_gray)
                        if len(cys) == 1:
                            np.save(name, nparr)
                            cv2.imwrite(name+'_slice.jpg', img_gray)
                            cv2.imwrite(name+'_mask.png', mask)                 # should not save as jpg since compression may change values
                        else:
                            np.save(f'{name}_{index}', nparr)
                            cv2.imwrite(f'{name}_slice_{index}.jpg', img_gray)
                            cv2.imwrite(f'{name}_mask_{index}.png', mask)       # should not save as jpg since compression may change values
                        print('slice saved')
                    # cv2.imshow("Segmented Object", mask)
                    # cv2.waitKey(0)