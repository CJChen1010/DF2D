import os
from os import path
from code_lib.Segmentation_210913 import Segmentor2D
from code_lib.Assignment_201020 import *
from skimage.io import imread
import numpy as np, pandas as pd
import multiprocessing
from functools import partial
from matplotlib.colors import ListedColormap

def mask2centroid(maskImg, ncore = 8):
    # ranges = np.split(np.arange(1, maskImg.max() + 1), ncore)
    ranges = np.split(np.arange(1, maskImg.max() + 1), np.linspace(1, maskImg.max() + 1, ncore+2).astype(int)[1:-1])

    pool = multiprocessing.Pool(ncore)
    f = partial(mask2centroid_parallel, mimg = maskImg)
    cent_arrs = list(pool.map(f, ranges))
    return np.concatenate(cent_arrs) 
    # centroids = []
    # for i in range(1, maskImg.max() + 1):
    #     xs, ys = np.where(maskImg == i)
    #     xc, yc = xs.mean().astype(int), ys.mean().astype(int)
    #     centroids.append((xc, yc))
    # return np.array(centroids)

def mask2centroid_parallel(rng, mimg):
    cent = []
    for i in rng:
        xs, ys = np.where(mimg == i)
        xc, yc = xs.mean().astype(int), ys.mean().astype(int)
        cent.append((xc, yc))    
    return np.array(cent)
    
nuc_path = '../3_background_subtracted/stitched/MIP_9_DRAQ5_ch00.tif'
dt_path = '../3_background_subtracted/stitched/MIP_0_anchor_ch02.tif'
n9_path = '../3_background_subtracted/stitched/MIP_0_anchor_ch03.tif'

saving_path = '../5_CellAssignment'

suff = '_DRAQ5'

bcmag = 'bcmag0.9'

    
spot_file = '../4_Decoded/output_Starfish/{}/all_spots_filtered.tsv'.format(bcmag)

if not path.exists(saving_path):
    os.makedirs(saving_path)

nuc_img = imread(nuc_path)
dt_img = imread(dt_path)
n9_img = imread(n9_path)

# Subtracting the dt_image from n9 and draq5
n9_sub = np.clip(n9_img.astype(int) - 3 * dt_img.astype(int), 0, 255).astype(np.uint8)
dq_sub = np.clip(nuc_img.astype(int) - 3 * dt_img.astype(int), 0, 255).astype(np.uint8)

# segmenting the nuclear image
segmentor = Segmentor2D()
mask = segmentor.segment_nuclei([nuc_img], diameters = 33, 
                         out_files = [path.join(saving_path, 'segmentation_mask{}.npy'.format(suff))])[0]
mask = np.load(path.join(saving_path, 'segmentation_mask{}.npy'.format(suff)))

# mask = segmentor.segment_cyto(nuc_imgs = [nuc_img], cyto_imgs = [dt_img], diameters = 40, 
#                          out_files = [path.join(saving_path, 'segmentation_mask_cyto.npy')])[0]
# mask = np.load(path.join(saving_path, 'segmentation_mask_cyto.npy'))

# mask = segmentor.segment_cyto(nuc_imgs = [dq_sub], cyto_imgs = [n9_sub], diameters = 40, 
#                          out_files = [path.join(saving_path, 'segmentation_mask_cyto_subt.npy')])[0]
# mask = np.load(path.join(saving_path, 'segmentation_mask_cyto_subt.npy'))

myCmap = np.random.rand(np.max(mask) + 1, 4)
myCmap[:, -1] = 1
myCmap[0] = (0, 0, 0, 1)
myCmap = ListedColormap(myCmap)

plt.figure(figsize = (int(mask.shape[0]/200), int(mask.shape[1]/200)))
plt.imshow(mask, cmap = myCmap)
plt.savefig(os.path.join(saving_path, 'mask.png'), dpi = 500, bbox_inches='tight')

# Rolony assignment
spot_df = pd.read_csv(spot_file, index_col=0, sep = '\t')
assigner = RolonyAssigner(nucleiImg=mask, rolonyDf=spot_df, axes = ['yg', 'xg'])
labels, dists = assigner.getResults()

spot_df['nucleus_label'] = labels
spot_df['dist2nucleus'] = np.round(dists, 2)
spot_df = spot_df.sort_values('nucleus_label', ignore_index = True)
spot_df.to_csv(path.join(saving_path, 'spots_assigned{}.tsv'.format(suff)), sep = '\t', index = False, float_format='%.3f')


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
centroids = mask2centroid(mask, ncore = 8)
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
nuc_gene_df = spot_df[['cell_label', 'gene']].groupby(by = ['cell_label', 'gene'], as_index = False).size()
nuc_gene_df = nuc_gene_df.reset_index().pivot(index = 'cell_label', columns = 'gene', values = 'size').fillna(0).astype(int)
nuc_gene_df.to_csv(path.join(saving_path, 'cell-by-gene{}.tsv'.format(suff)), sep = '\t')

