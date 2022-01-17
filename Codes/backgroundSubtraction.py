import os, re, numpy as np
import skimage.io as io
from skimage.filters import threshold_otsu as otsu
from matplotlib import pyplot as plt
import yaml
import argparse
from tifffile import imsave

parser = argparse.ArgumentParser()
parser.add_argument('param_file')
args = parser.parse_args()
params = yaml.safe_load(open(args.param_file, "r"))

bgRounds = params['background_cycles']
datadir = params['reg_dir']
savedir = params['background_subt_dir']
channels = params["subtraction_channels"]


for fov in sorted(os.listdir(datadir)):
    if not fov.startswith('FOV'):
        continue
    if not os.path.isdir(os.path.join(savedir , fov)):
        os.makedirs(os.path.join(savedir , fov))
        
    for ch in channels:
        bcgImgs = [io.imread('{3}/{0}/MIP_{2}_{0}_{1}.tif'.format(fov, ch, rnd, datadir)) for rnd in bgRounds]
        bcgImg = np.max(bcgImgs, axis = 0)
        # thresh = otsu(bcgImg)/1.5  # 25
        # bcgMask = bcgImg > thresh
        for img_file in os.listdir(os.path.join(datadir, fov)):
            if (ch in img_file) and ('.tif' in img_file):
                img = io.imread(os.path.join(datadir, fov, img_file))
                # img[bcgMask] = 0
                img = np.clip(img.astype(int) - bcgImg.astype(int), 0, None).astype(img.dtype)
                # io.imsave(os.path.join(savedir , fov, img_file), img)
                imsave(os.path.join(savedir , fov, img_file), img)
    
    # move the brightfield channel
    for img_file in os.listdir(os.path.join(datadir, fov)):
        if ('ch01' in img_file) and ('.tif' in img_file):
            img = io.imread(os.path.join(datadir, fov, img_file))
            io.imsave(os.path.join(savedir , fov, img_file), img)