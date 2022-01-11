""" This combines the spot tables from all FOVs. While combining, it removes duplicate spots
    from overlapping tiles and filters spots based on their distance-to-barcode measure and 
    the empty-barcode rate. This whole process is repeated for all bcmags coming from StarFish.
    The output is written in the same directory that the spot tables are read from.
"""
import os, re, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET

def getMetaData(metadataXml):
    """ parses a metadata .xml file and returns 
        1) size of FOVs in pixels
        2) physical dimension of the voxels
        3) #FOVs
    """ 
    tree = ET.parse(metadataXml)
    root = tree.getroot()
    dimDscrpt = [item for item in root.findall("./Image/ImageDescription/Dimensions/DimensionDescription")]
    # idDict = ['1' : 'X', '2' : 'Y', '3' : 'Z']
    n_pixels = {dimInfo.attrib['DimID']: int(dimInfo.attrib['NumberOfElements']) for dimInfo in dimDscrpt if dimInfo.attrib['DimID'] in ['1', '2', '3']}
    voxelSizes = {}
    for dimInfo in dimDscrpt: 
        if dimInfo.attrib['DimID'] not in ['1', '2', '3']:
            continue
        unit = dimInfo.attrib['Unit']
        if unit == 'um':
            scaleFac = 1
        elif unit == 'mm':
            scaleFac = 10 ** 3
        elif unit == 'm':
            scaleFac = 10 ** 6
        voxelSizes[dimInfo.attrib['DimID']] = scaleFac * float(dimInfo.attrib['Length']) / n_pixels[dimInfo.attrib['DimID']]


    tiles = [tile for tile in root.findall("./Image/Attachment/Tile")]

    return n_pixels, voxelSizes, len(tiles)

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
    spot_df_trimmed = spot_df.loc[spot_df['cum_empty_frac'] <= cutoff]
    spot_df_highDist = spot_df.loc[spot_df['cum_empty_frac'] > cutoff]
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


decoding_dir = '../4_Decoded/output_Starfish' # the main directory for decoding 
bcmags = ["bcmag0.9"]
# bcmags = [file for file in os.listdir(decoding_dir) 
#           if os.path.isdir(os.path.join(decoding_dir, file)) 
#               and 'bcmag' in file]  # all the bcmags that were used for the experiment

# VOXEL = {"Y":0.290, "X":0.290, "Z":0.420}  # voxel size info
RND_ALIGNED = "5_dc3"
metadataFile = "../1_Projected/MetaData/{0}.xml".format(RND_ALIGNED)
npix, vox, number_of_fovs = getMetaData(metadataFile)

VOXEL = {"Y":vox['2'], "X":vox['1'], "Z":vox['3']}


emptyFractionThresh = 0.15 # finding a distance that this fraction of decoded spots with smaller distances are empty
fov_pat = r"FOV(\d+)" # the regex showing specifying the tile names. 
overlapRemovalRadius = 4.5 # radius in pixels for removing overlaps

for bcmag in bcmags: 
    print("filtering barcode magnitude: {}".format(bcmag))
    all_files = [os.path.join(decoding_dir, bcmag, file)
                 for file in os.listdir(os.path.join(decoding_dir, bcmag))
                 if re.search(fov_pat, file)]
    all_files.sort(key = lambda x: int(re.search(fov_pat, x).group(1)))
    filtered_spots, overlapFree_spots, removed_spots = makeSpotTable(all_files, emptyFractionThresh, VOXEL, removeRadius=overlapRemovalRadius)
    filtered_spots.reset_index(drop = True).to_csv(os.path.join(decoding_dir, bcmag, 'all_spots_filtered.tsv'), sep = '\t')
    
    overlapFree_spots.reset_index(drop = True).to_csv(os.path.join(decoding_dir, bcmag, 'all_spots_overlapFree.tsv'), sep = '\t')
    
    removed_spots = removed_spots.loc[removed_spots['gene'] != 'Empty']
    removed_spots.reset_index(drop = True).to_csv(os.path.join(decoding_dir, bcmag, 'all_removed_spots.tsv'), sep = '\t')

    plt.figure()
    plt.plot(overlapFree_spots['distance'], overlapFree_spots['cum_empty_frac'], '*')
    plt.xlabel("barcode distance")
    plt.ylabel("empty fraction")
    plt.savefig(os.path.join(decoding_dir, bcmag, 'distance_emptyRate_plot.png'))