#!/usr/bin/env python3

import os
from os.path import isfile, isdir
import sys

if __name__ == '__main__':
    dir_name = sys.argv[1]
    os.chdir(dir_name)

    dirs = os.listdir('.')
    for current_dir_name in dirs:
        if not isdir(current_dir_name):
            continue

        print()
        print('Renaming files in {0}'.format(current_dir_name))
        os.chdir(current_dir_name)
        files = os.listdir('.')
        img_counter = 0
        obj_name = current_dir_name
        for f in files:
            if not isfile(f):
                continue
            print('Renaming {0}'.format(f))
            new_file_name = '{0}_{1}.jpg'.format(obj_name, img_counter)
            os.rename(f, new_file_name)

            if not isdir('object_masks'):
                continue

            os.chdir('object_masks')
            f_name, _ = f.split('.')

            print('Renaming object mask {0}'.format(f_name))
            mask_name = '{0}_mask.jpg'.format(f_name)
            if not isfile(mask_name):
                os.chdir('..')
                continue

            new_mask_name = '{0}_{1}_mask.jpg'.format(obj_name, img_counter)
            os.rename(mask_name, new_mask_name)
            os.chdir('..')

            img_counter += 1
        os.chdir('..')
