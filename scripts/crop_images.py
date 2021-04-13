import argparse
import os
import glob
import numpy as np
import cv2

parser = argparse.ArgumentParser()
# Common arguments
parser.add_argument('image_dir', type=str)
parser.add_argument('output_dir', type=str)
args = parser.parse_args()

image_dir = args.image_dir
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_list = glob.glob(image_dir + '/*.png')
for filename in image_list:
    print(f'Processing: {filename}')
    img = cv2.imread(filename)

    h, w, _ = img.shape
    h -= h % 64
    w -= w % 64
    img = img[:h, :w, :]

    output_fn = os.path.join(output_dir, os.path.basename(filename))
    print(f'Writing: {output_fn}')
    cv2.imwrite(output_fn, img)
