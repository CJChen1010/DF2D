import os, re
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log 
from skimage.io import imread
import matplotlib.collections as col

def normalize(im, ceil):
    return (np.clip(im / ceil * 255, a_min=0, a_max = 255)).astype(np.uint8)

def fov_changer(fov, ndigit = 3):
    # changing FOVXX to FOVXXX
    num = int(fov[3:])
    return("FOV{:03d}".format(num))

savingdir = "./5_Analysis"
rolonyPlotDir = os.path.join(savingdir, "RolPlots")

img_dir = "../3_background_subtracted"
spot_addr = "../4_Decoded/output_Starfish_maxArea25/bcmag0.9/all_spots_filtered.tsv"

max_int = 50 # normalizing the image to this maximum
anchor_pat = r"MIP_0_anchor_{fov}_ch00.tif"
fovs = [file for file in os.listdir(img_dir) if file.startswith('FOV')]

if not os.path.isdir(savingdir):
    os.mkdir(savingdir)
if not os.path.isdir(rolonyPlotDir):
    os.mkdir(rolonyPlotDir)
# img_dir = "/media/Scratch_SSD_Voyager/Kian/DART-FISH/211013_PermeabilizationTest/Decode_K2100233_1-D/3_background_subtracted/"

# spot_addr = "/media/Scratch_SSD_Voyager/Kian/DART-FISH/211013_PermeabilizationTest/Decode_K2100233_1-D/4_Decoded/output_Starfish_maxArea25/bcmag0.9/all_spots_filtered.tsv"

# img_dir = "/media/Scratch_SSD_Voyager/Kian/DART-FISH/211112_PermeabilizationTest2/Decode-K2100013_3-B/3_background_subtracted/"

# spot_addr = "/media/Scratch_SSD_Voyager/Kian/DART-FISH/211112_PermeabilizationTest2/Decode-K2100013_3-B/4_Decoded/output_Starfish/bcmag0.9/all_spots_filtered.tsv"


spot_df = pd.read_csv(spot_addr, sep = "\t", index_col = 0)
spot_df = spot_df.loc[spot_df['gene'] != 'Empty']


n_blobs = []
n_rols = []
all_blobs = []
for fov in fovs: 
    img_addr = os.path.join(img_dir, fov, anchor_pat.format(fov=fov))
    anc_img = imread(img_addr)
    this_blb = blob_log(normalize(anc_img, max_int), min_sigma=0.7, 
                        max_sigma=2, num_sigma=6, overlap = 0.6, threshold = 0.3)
    all_blobs.append(this_blb)
    n_blobs.append(this_blb.shape[0])
    n_rols.append(spot_df.loc[spot_df['fov'] == fov_changer(fov, 3)].shape[0])


n_rols, n_blobs = np.array(n_rols), np.array(n_blobs)
p = np.polyfit(n_rols, y = n_blobs, deg=1 )
plt.plot([0, np.max(n_rols)], [0, np.max(n_rols)], c = 'green', alpha = 0.5, label = 'x=y line')
plt.plot([0, np.max(n_rols)], np.polyval(p, [0, np.max(n_rols)]), c = 'orange', alpha = 0.5, label = 'fitted line')
plt.scatter(n_rols, n_blobs, alpha =0.7)
plt.xlabel("#decoded rolonies")
plt.ylabel("#anchor blobs")
plt.legend()
plt.savefig(os.path.join(savingdir, "decoded_rolonies-v-anchor_blobs.pdf"))


dec_rate = n_rols / n_blobs
fig, ax = plt.subplots()
ax.hist(dec_rate, bins = 20, alpha = 0.8)
ax.vlines(np.median(dec_rate), *ax.get_ylim(), color = 'orange', label = 'median')
ax.set_title("Decoding Rate Histogram")
ax.set_xlabel("decoding rate")
ax.set_ylabel("frequency")
ax.legend()
plt.savefig(os.path.join(savingdir, "decoding_rate_histogram.pdf"))


""" Plotting the rolonies, blobs and decoded rolonies"""
for i, fov in enumerate(fovs): 
    img_addr = os.path.join(img_dir, fov, anchor_pat.format(fov=fov))
    anc_img = imread(img_addr)
    this_blb = all_blobs[i]
    spot_df_fov = spot_df.loc[spot_df['fov'] == fov]
    fig, ax = plt.subplots(figsize = (20, 20))
    this_blb = blob_log(normalize(anc_img, max_int), min_sigma=0.7, 
                        max_sigma=2, num_sigma=6, overlap = 0.6, threshold = 0.3)

    ax.imshow(normalize(anc_img, max_int), cmap = 'gray',alpha=0.9)
    circ_patches = []
    for x, y, r in this_blb:
        circ = plt.Circle((y, x), 3 * r, color = 'red', fill = None, alpha=0.8)
        circ_patches.append(circ)

    # add the circles as a collection of patches (faster)
    col1 = col.PatchCollection(circ_patches, match_original=True)
    ax.add_collection(col1)

    spot_df_fov = spot_df.loc[spot_df['fov'] == fov]
    circ_patches = []
    for i, row in spot_df_fov.iterrows():
        circ = plt.Circle((row['x'], row['y']), 3, color = 'lightgreen', fill = None, alpha=0.8)
        circ_patches.append(circ)

    # add the circles as a collection of patches (faster)
    col2 = col.PatchCollection(circ_patches, match_original=True)
    ax.add_collection(col2)
    
    plt.tight_layout()
    fig.savefig(os.path.join(rolonyPlotDir, "{}.png".format(fov)))



""" Rolony Stats """
def getHistHeight(histOut, x):
    n, bins, patches = histOut
    bar_n = np.where(x <= bins)[0][0] - 1
    h = n[bar_n]
    return h

cellbygene = spot_df[['cell_label', 'gene']].groupby(by = ['cell_label', 'gene'], as_index = False).size()
cellbygene = cellbygene.reset_index().pivot(index = 'cell_label', columns = 'gene', values = 'size').fillna(0).astype(int)


n_cells = cellbygene.index[-1]
countpercell = cellbygene.sum(axis=1)
gene_per_nuc = (cellbygene > 0).sum(axis=1)
max_x_rols = np.floor(np.percentile(countpercell, 99))
max_x_genes = np.floor(np.percentile(gene_per_nuc, 99))

fig, ax = plt.subplots(ncols = 2, nrows=2, figsize = (12, 10))
ax = ax.flatten()

""" Rolonies per cell"""
O = ax[0].hist(countpercell, bins = 40, density=False, alpha = 0.8)
ax[0].set_title('#rolony count per cell histogram')
ax[0].set_yscale('log')
ax[0].vlines(np.mean(countpercell), ymin = 0, ymax = getHistHeight(O, np.mean(countpercell)), 
             color = 'red', label = "mean", linestyles='dashed')
ax[0].legend()


bins = np.arange(1, max_x_rols, 1)
count_cum = [np.sum(countpercell >= i) for i in bins] / n_cells
bins = np.append(0, bins) # have to append 0 and 1 since cells with no rolony are not in the count matrix
count_cum = np.append(1, count_cum)

ax[1].bar(bins, 100 * count_cum, width = 1, alpha = 0.8)
ax[1].set_yscale('log')
ax[1].set_xticks(np.arange(0, bins[-1], step = 10))
ax[1].vlines(x = bins[5], ymin = 0, ymax= 100 * count_cum[5], 
             colors = 'red', linestyles='dashed', 
             label = 'at least 5 rolonies')
ax[1].vlines(x = bins[20], ymin = 0, ymax= 100 * count_cum[20], 
             colors = 'orange', linestyles='dashed', 
             label = 'at least 20 rolonies')
ax[1].set_title('Cumulative rolony count per nucleus')
ax[1].set_ylabel('% cells')
ax[1].set_xlabel('At least #rolony per cell')
ax[1].legend()

""" Genes per cell"""
bins = np.arange(1, max_x_genes, 1)
gene_cum = [np.sum(gene_per_nuc >= i) for i in bins] / n_cells
bins = np.append(0, bins)
gene_cum = np.append(1, gene_cum)

O = ax[2].hist(gene_per_nuc, density=False, alpha = 0.8, bins=40)
ax[2].set_title('#Genes per cell distribution')
ax[2].set_yscale('log')
ax[2].vlines(np.mean(gene_per_nuc), ymin = 0, ymax = getHistHeight(O, np.mean(gene_per_nuc)), 
             color = 'red', label = "mean", linestyles='dashed')
ax[2].legend()

ax[3].bar(bins, 100 * gene_cum, width = 1, alpha = 0.8)
ax[3].set_yscale('log')
ax[3].set_xticks(np.arange(0, bins[-1], 5))
ax[3].vlines(x = bins[5], ymin = 0, ymax= 100 * gene_cum[5], 
             colors = 'red', linestyles='dashed', 
             label = 'at least 5 genes')
ax[3].vlines(x = bins[20], ymin = 0, ymax= 100 * gene_cum[20], 
             colors = 'orange', linestyles='dashed', 
             label = 'at least 20 genes')

ax[3].set_title('Cumulative #genes count per nucleus')
ax[3].set_ylabel('%cells')
ax[3].set_xlabel('at least #gene per cell')
ax[3].legend()


plt.tight_layout()
plt.savefig(os.path.join(savingdir, 'rolony_stats.png'), dpi=200)

