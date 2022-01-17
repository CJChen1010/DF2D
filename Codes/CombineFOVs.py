""" This combines the spot tables from all FOVs. While combining, it removes duplicate spots
    from overlapping tiles and filters spots based on their distance-to-barcode measure and 
    the empty-barcode rate. This whole process is repeated for all bcmags coming from StarFish.
    The output is written in the same directory that the spot tables are read from.
"""
import os, re, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
from utils import getMetaData
import yaml
import argparse

def removeOverlapRolonies(rolonyDf, x_col = 'x', y_col = 'y', removeRadius = 5.5):
    """ For each position, find those rolonies that are very close to other rolonies 
        in other positions and remove them.
        x_col and y_col are the names of the columns for x and y coordinates.
        removeRadius is in any unit that x_col and y_col are.
    """
    geneList = rolonyDf.target.unique()
    reducedRolonies = []
    for gene in geneList:
        thisGene_rolonies = rolonyDf.loc[rolonyDf.target == gene]
        for pos in sorted(rolonyDf['fov'].unique()):
            thisPos = thisGene_rolonies.loc[thisGene_rolonies['fov'] == pos]
            otherPos = thisGene_rolonies.loc[thisGene_rolonies['fov'] != pos]
            if (len(thisPos) <= 0 ) or (len(otherPos) <= 0 ):
                continue
            nnFinder = cKDTree(thisPos[[x_col, y_col]])
            nearestDists, nearestInds = nnFinder.query(otherPos[[x_col, y_col]], distance_upper_bound = removeRadius)
            toRemoveFromThisPos_index = thisPos.index[nearestInds[nearestDists < np.inf]]
            thisGene_rolonies = thisGene_rolonies.drop(toRemoveFromThisPos_index)
        reducedRolonies.append(thisGene_rolonies)
    return pd.concat(reducedRolonies) 


def filterByEmptyFraction(spot_df, cutoff):
    spot_df = spot_df.sort_values('distance')
    spot_df['isEmpty'] = spot_df['target'].str.startswith('Empty')
    spot_df['cum_empty'] = spot_df['isEmpty'].cumsum()
    spot_df['cum_empty_frac'] = spot_df['cum_empty'] / np.arange(1, spot_df.shape[0] + 1)
    cut_ind = np.where(spot_df['cum_empty_frac'] <= cutoff)[0][-1]
    spot_df_trimmed = spot_df.iloc[:cut_ind]
    spot_df_highDist = spot_df.iloc[cut_ind:]
    return spot_df_trimmed, spot_df_highDist, spot_df



def makeSpotTable(files_paths, emptyFractionCutoff, voxel_info, removeRadius=5.5):
    # Concatenating spots from all FOVs and converting the physical coordinates to pixels 
    allspots = []
    for file in all_files: 
        thisSpots = pd.read_csv(file, index_col = 0)
        thisSpots['xg'] = (round(thisSpots['xc'] / voxel_info['X'])).astype(int)
        thisSpots['yg'] = (round(thisSpots['yc'] / voxel_info['Y'])).astype(int)
        thisSpots['zg'] = (round(thisSpots['zc'] / voxel_info['Z'])).astype(int)
        thisSpots['fov'] = re.search(fov_pat, file).group()
        thisSpots['spot_id'] = thisSpots['fov'] + '_' + thisSpots['spot_id'].astype(str)
        allspots.append(thisSpots)

    allspots = pd.concat(allspots, ignore_index=True)

    allspots['gene'] = allspots['target'].str.extract(r"^(.+)_")

    # removing empty spots of area 1 if they exist
    allspots = allspots.loc[~((allspots['gene'] == 'Empty') & (allspots['area'] == 1))]

    allspots = allspots.sort_values('distance')

    # Removing duplicate rolonies caused the overlapping regions of FOVs
    allspots_reduced = removeOverlapRolonies(allspots, x_col='xg', y_col = 'yg', removeRadius=removeRadius)

    # Keeping only spots with small distance to barcode so that `emptyFractionThresh` of spots are empty.
    allspots_trimmed, allspots_highDist, allspots_reduced = filterByEmptyFraction(allspots_reduced, cutoff = emptyFractionCutoff)

    return allspots_trimmed, allspots_reduced, allspots_highDist

parser = argparse.ArgumentParser()
parser.add_argument('param_file')
args = parser.parse_args()
params = yaml.safe_load(open(args.param_file, "r"))

decoding_dir = params['dc_out'] # the main directory for decoding 
bcmags = ["bcmag{}".format(params['bcmag'])]

metadataFile = os.path.join(params['dir_data_raw'], params['ref_reg_cycle'], 'MetaData', "{}.xml".format(params['ref_reg_cycle']))
npix, vox, number_of_fovs = getMetaData(metadataFile)

VOXEL = {"Y":vox['2'], "X":vox['1'], "Z":vox['3']}


emptyFractionThresh = params['emptyFractionThresh'] # finding a distance that this fraction of decoded spots with smaller distances are empty
fov_pat = params['fov_pat'] # the regex showing specifying the tile names. 
overlapRemovalRadius = params['overlapRemovalRadius'] # radius in pixels for removing overlaps

for bcmag in bcmags: 
    print("filtering barcode magnitude: {}".format(bcmag))
    savingpath = decoding_dir + "_" + bcmag
    all_files = [os.path.join(savingpath, file)
                 for file in os.listdir(os.path.join(savingpath))
                 if re.search(fov_pat, file)]
    all_files.sort(key = lambda x: int(re.search(fov_pat, x).group(1)))
    filtered_spots, overlapFree_spots, removed_spots = makeSpotTable(all_files, emptyFractionThresh, VOXEL, removeRadius=overlapRemovalRadius)
    filtered_spots.reset_index(drop = True).to_csv(os.path.join(savingpath, 'all_spots_filtered.tsv'), sep = '\t')
    
    overlapFree_spots.reset_index(drop = True).to_csv(os.path.join(savingpath, 'all_spots_overlapFree.tsv'), sep = '\t')
    
    removed_spots = removed_spots.loc[removed_spots['gene'] != 'Empty']
    removed_spots.reset_index(drop = True).to_csv(os.path.join(savingpath, 'all_removed_spots.tsv'), sep = '\t')

    fig, axes = plt.subplots(ncols = 2, figsize = (10, 6))
    axes[0].plot(np.arange(0, overlapFree_spots.shape[0]), overlapFree_spots['cum_empty_frac'])
    axes[0].set_xlabel("spot index")
    axes[0].set_ylabel("empty fraction")

    axes[1].plot(np.arange(0, overlapFree_spots.shape[0]), overlapFree_spots['distance'])
    axes[1].set_xlabel("spot number")
    axes[1].set_ylabel("barcode distance")
    plt.tight_layout()
    plt.savefig(os.path.join(savingpath, 'distance_emptyRate_plot.png'))