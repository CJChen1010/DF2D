import os
from os import path
from code_lib.Segmentation_210913 import Segmentor2D
from code_lib.Assignment_201020 import *
from skimage.io import imread
import numpy as np, pandas as pd
import multiprocessing
from functools import partial
from matplotlib.colors import ListedColormap
import yaml
import argparse

def mask2centroid(maskImg, ncore = 8):
    # ranges = np.split(np.arange(1, maskImg.max() + 1), ncore)
    ranges = np.split(np.arange(1, maskImg.max() + 1), np.linspace(1, maskImg.max() + 1, ncore+2).astype(int)[1:-1])

    pool = multiprocessing.Pool(ncore)
    f = partial(mask2centroid_parallel, mimg = maskImg)
    cent_arrs = list(pool.map(f, ranges))
    return np.concatenate(cent_arrs) 

def mask2centroid_parallel(rng, mimg):
    cent = []
    for i in rng:
        xs, ys = np.where(mimg == i)
        xc, yc = xs.mean().astype(int), ys.mean().astype(int)
        cent.append((xc, yc))    
    return np.array(cent)
    
parser = argparse.ArgumentParser()
parser.add_argument('param_file')
args = parser.parse_args()
params = yaml.safe_load(open(args.param_file, "r"))

stitch_dir = params['stitch_dir']

if 'nuc' in params['segmentation_type']:
    nuc_path = os.path.join(stitch_dir, "MIP_{}_{}.tif".format(params['nuc_rnd'], params['nuc_ch']))
    nuc_img = imread(nuc_path)

if 'cyto' in params['segmentation_type']: 
    cyto_path = os.path.join(stitch_dir, "MIP_{}_{}.tif".format(params['cyto_rnd'], params['cyto_ch']))
    cyto_img = imread(cyto_path)

saving_path = params['seg_dir']
if not path.exists(saving_path):
    os.makedirs(saving_path)

suff = params['seg_suf']

bcmag = params['bcmag']
    
spot_file = os.path.join(params['dc_out'] + '_' + bcmag, 'all_spots_filtered.tsv')


# segmenting the nuclear image
diam = params['seg_diam']
segmentor = Segmentor2D()

if params['segmentation_type'] == 'nuc':
    mask = segmentor.segment_nuclei([nuc_img], diameters = diam, 
                         out_files = [path.join(saving_path, 'segmentation_mask{}.npy'.format(suff))])[0]
elif params['segmentation_type'] == 'cyto':
    mask = segmentor.segment_cyto([cyto_img], diameters = diam, 
                         out_files = [path.join(saving_path, 'segmentation_mask{}.npy'.format(suff))])[0]
elif params['segmentation_type'] == 'cyto+nuc':
    mask = segmentor.segment_cyto_nuc([nuc_img], [cyto_img], diameters = diam, 
                         out_files = [path.join(saving_path, 'segmentation_mask{}.npy'.format(suff))])[0]

# plot segmentation mask
myCmap = np.random.rand(np.max(mask) + 1, 4)
myCmap[:, -1] = 1
myCmap[0] = (0, 0, 0, 1)
myCmap = ListedColormap(myCmap)

plt.figure(figsize = (int(mask.shape[0]/200), int(mask.shape[1]/200)))
plt.imshow(mask, cmap = myCmap)
plt.savefig(os.path.join(saving_path, 'mask{}.png'.format(suff)), dpi = 500, bbox_inches='tight')


# Rolony assignment
spot_df = pd.read_csv(spot_file, index_col=0, sep = '\t')
assigner = RolonyAssigner(nucleiImg=mask, rolonyDf=spot_df, axes = ['yg', 'xg'])
labels, dists = assigner.getResults()

spot_df['cell_label'] = labels
spot_df['dist2cell'] = np.round(dists, 2)
spot_df = spot_df.sort_values('cell_label', ignore_index = True)
spot_df.to_csv(path.join(saving_path, 'spots_assigned{}.tsv'.format(suff)), sep = '\t', index = False, float_format='%.3f')


# plotting assigned rolonies
print("plotting assigned rolonies")
fig = plt.figure(figsize = (int(mask.shape[0]/200), int(mask.shape[1]/200)))
ax = fig.gca()
plotRolonies2d(spot_df, mask, coords = ['xg', 'yg'], label_name='cell_label', ax = ax, backgroudImg=nuc_img, backgroundAlpha=0.6)
fig.savefig(path.join(saving_path, 'assigned_rolonies{}.png'.format(suff)),
            transparent = False, dpi = 500, bbox_inches='tight')
print("plotting assigned rolonies done")


# finding the cells centroids
centroids = mask2centroid(mask, ncore = params['centroid_npool'])
centroid_df = pd.DataFrame({'cell_label' : np.arange(1, mask.max() + 1), 
                            'centroid_x' : centroids[:, 0], 'centroid_y' : centroids[:, 1]})
centroid_df.to_csv(path.join(saving_path, 'cell_locations{}.tsv'.format(suff)), sep = '\t', index = False)

# plotting the cells with their label
fig = plt.figure(figsize = (int(mask.shape[0]/200), int(mask.shape[1]/200)))
ax = fig.gca()
ax.imshow(nuc_img, cmap='gray')
ax.scatter(centroids[:, 1], centroids[:, 0], s = 1, c='red')
for i in range(centroids.shape[0]):
    ax.text(centroids[i, 1], centroids[i, 0], str(i), fontsize = 5, c = 'orange')
fig.savefig(path.join(saving_path, 'cell_map{}.png'.format(suff)),
            transparent = True, dpi = 400, bbox_inches='tight')

# Making the cell by gene matrix
spot_df = spot_df.loc[spot_df['dist2cell'] <= params['max_rol2nuc_dist']] # filtering rolonies based on distance to cell
nuc_gene_df = spot_df[['cell_label', 'gene']].groupby(by = ['cell_label', 'gene'], as_index = False).size()
nuc_gene_df = nuc_gene_df.reset_index().pivot(index = 'cell_label', columns = 'gene', values = 'size').fillna(0).astype(int)
nuc_gene_df.to_csv(path.join(saving_path, 'cell-by-gene{}.tsv'.format(suff)), sep = '\t')

