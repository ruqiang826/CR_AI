#!/usr/bin/env python

import sys,os,random

if __name__ == '__main__':
    input_dir = sys.argv[1]
    im_names = os.listdir(input_dir)

    for im_name in im_names:
        #im_file = os.path.join(input_dir, 'img', im_name)

        if random.randint(1,100) <= 10:
            index = im_name.rindex('.')
            f = im_name[:index] + ".*"
            print f,
    
