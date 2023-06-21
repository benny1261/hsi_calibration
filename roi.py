import os
import numpy as np
from spectral.io import envi
import spectral
import glob

'''
requires:
xxx.hdr
xxx.raw
xxx.txt(position info)
'''

OUTPUT_CHANNEL = 50

if __name__ == '__main__':
    rootdir = os.path.dirname(__file__)+r'/data'
    spectral.settings.envi_support_nonlowercase_params = 'TRUE'

    for root, dirs, files in os.walk(rootdir):
        for dir_name in dirs:
            current_folder = os.path.join(root, dir_name)
            print(current_folder)
            os.chdir(current_folder)
            hdr_list, raw_list, txt_list = glob.glob('*.hdr'), glob.glob('*.raw'), glob.glob('*.txt')
            hsi_dict = {}
            for _ in hdr_list:
                name = _.split(".")[0]
                try:
                    hsi_dict[name] = envi.open(name+ ".hdr" , name+ ".raw")     # our hsi metadata stored in ENVI raster format
                    arr = hsi_dict[name].load()
                    print(name,"-----------------")
                except:
                    print('cannot read hsi file of ', name)
            # nparr = arr.copy()