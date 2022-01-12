
# import immac, re, shutil, warnings, sys
import re, shutil, warnings, sys, os	# Kian: added 201011
from time import time
from datetime import datetime
from code_lib import TwoDimensionalAligner_2 as myAligner # Kian: added 201011
from os import chdir, listdir, getcwd, path, makedirs, remove,  walk
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from code_lib import tifffile as tiff # <http://www.lfd.uci.edu/~gohlke/code/tifffile.py> # Kian: added 201011
from multiprocessing import Pool 	# Kian: added 210323
import functools	# Kian: added 210323
import yaml	# Kian: added 220111
from code_lib.utils import # Kian: added 220111


def listdirectories(directory='.', pattern = '*'):	# Kian: added 201011
	'''
	listdirectories returns a list of all directories in path
	'''
	directories = []
	for i in next(walk(directory))[1]:
#		if match(pattern, i): 
			directories.append(i)
	directories.sort()
	return directories



def mip_gauss_tiled(fov, rnd, dir_root, dir_output='./MIP_gauss',
		 sigma=0.7, channel_int='ch00'):
	'''
	Modified from Matt Cai's MIP.py 
	Maximum intensity projection along z-axis
	'''
	# get current directory and change to working directory
	current_dir = getcwd()

	
	dir_parent = path.join(dir_root)
	chdir(dir_parent + "/" + rnd)
		
	#get all files for position for channel
	image_names = [f for f in listdir('.') if re.match(r'.*_s' + fov + r'.*_' + channel_int + r'\.tif', f)]

	# put images of correct z_range in list of array
	nImages = len(image_names)
	image_list = [None]*nImages
	for i in range(len(image_names)):
		image_list[i] = (ndimage.gaussian_filter(plt.imread(image_names[i]), sigma=sigma))
		  
	# change directory back to original
	chdir(current_dir)

	image_stack = np.dstack(image_list)
	
	max_array = np.amax(image_stack, axis=2)
	
	# PIL unable to save uint16 tif file
	# Need to use alternative (like libtiff)
	
	#Make directories if necessary
	if not dir_output.endswith('/'):
		dir_output = "{0}/".format(dir_output)
	#mkreqdir(dir_output)
	if not path.isdir(dir_output + 'FOV{}'.format(fov)):
		makedirs(dir_output + 'FOV{}'.format(fov))
		
	#Change to output dir and save MIP file
	chdir(dir_output)
	tiff.imsave('FOV{}'.format(fov) + '/MIP_' + rnd + '_FOV{}'.format(fov) + '_' + channel_int + '.tif', max_array)
	
	# change directory back to original
	chdir(current_dir)

def alignFOV(fov, out_mother_dir, round_list, mip_dir, channel_DIC, cycle_other, channel_DIC_other, reference_cycle):
	print(datetime.now().strftime("%Y-%d-%m_%H:%M:%S: {} started to align".format(fov)))

	""" mip_dir has to be a relative path, not an absolute path, i.e. "../"""
	init_dir = os.getcwd()
	out_dir = os.path.join(out_mother_dir, fov)
	in_dir = os.path.join(mip_dir, fov)

	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	os.chdir(out_dir)

	for rnd in round_list:
		aligner = myAligner.TwoDimensionalAligner(
				destinationImagesFolder = os.path.join(init_dir, in_dir), 
				originImagesFolder = os.path.join(init_dir, in_dir),
				originMatchingChannel = channel_DIC if rnd not in cycle_other else channel_DIC_other[rnd],
				destinationMatchingChannel = channel_DIC_reference, 
				imagesPosition = fov, 
				destinationCycle = reference_cycle,
				originCycle = rnd,
				resultDirectory = "./",
				MaximumNumberOfIterations = 400)

	os.chdir(init_dir)

params = yaml.safe_load(open("./params.yaml", "r"))

#Raw Data Folder
dir_data_raw = params['dir_data_raw']

#Where MIPs are written to
dir_output = params['proj_dir']
if not path.isdir(dir_output):
	makedirs(dir_output)

#Where Aligned MIPs are written to
dir_output_aligned = params['reg_dir']
if not path.isdir(dir_output_aligned):
	makedirs(dir_output_aligned)

#rounds
rnd_list = params['reg_rounds']

#sigma for gaussian blur
sigma = params['sigma']

reference_cycle = params['ref_reg_cycle'] # Which cycle to align to
channel_DIC_reference = params['ref_reg_ch'] # DIC channel for reference cycle
channel_DIC = channel_DIC_reference # DIC channel for (non-reference) decoding cycles (the channel we use for finding alignment parameters)
cycle_other = list(params['cycle_other']) # if there are other data-containing folders which need to be aligned but are not names "CycleXX"
channel_DIC_other = params['cycle_other'] # DIC channel for otPher data-containing folders

#Number of FOVs
metadataFile = os.path.join(params['dir_data_raw'], reference_cycle, 'MetaData', "{}.xml".format(reference_cycle))
_, _, n_fovs = getMetaData(metadataFile)

n_pool = params['reg_npool']

t0 = time()

#MIP
for rnd in rnd_list:
	if ("DRAQ5" in rnd):
		channel_list = [0, 1]
	else:
		channel_list = [0, 1, 2, 3]
	

	for channel in channel_list:
		print('Generating MIPs for ' + rnd + ' channel {0} ...'.format(channel))

		# defining a partial function from mip_gauss_tiled that only take fov as input
		mip_gauss_partial = functools.partial(mip_gauss_tiled, rnd=rnd, dir_root=dir_data_raw, 
			dir_output=dir_output, sigma=sigma, channel_int="ch0{0}".format(channel))
		with Pool(8) as p:
			list(p.map(mip_gauss_partial, [str(f).zfill(len(str(n_fovs))) for f in range(n_fovs)]))

		print('Done\n')

		# move the metadata file to the output directory
		metaFile = path.join(dir_data_raw, rnd, 'MetaData', "{0}.xml".format(rnd))
		if path.isfile(metaFile):
			if not path.exists(path.join(dir_output, "MetaData")):
				os.mkdir(path.join(dir_output, "MetaData"))

			shutil.copy2(src = metaFile, 
				dst = path.join(dir_output, 'MetaData', "{0}.xml".format(rnd)))
		else:
			print("MetaData file wasn't found at {}".format(path.join(dir_data_raw, rnd, 'MetaData', "{0}.xml".format(rnd))))
	
t1 = time()

print('Elapsed time ', t1 - t0)



position_list = [d for d in listdirectories(path.join(dir_output)) if d.startswith('FOV')]
#Align
currentTime = datetime.now() 
reportFile = path.join(dir_output_aligned, currentTime.strftime("%Y-%d-%m_%H:%M_SITKAlignment.log"))
sys.stdout = open(reportFile, 'w') # redirecting the stdout to the log file


partial_align = functools.partial(alignFOV, out_mother_dir=dir_output_aligned, round_list=rnd_list, 
	mip_dir=dir_output, channel_DIC=channel_DIC, cycle_other=cycle_other, 
	channel_DIC_other=channel_DIC_other, reference_cycle=reference_cycle)

with Pool(n_pool) as P:
	list(P.map(partial_align, position_list))
print("done")

		
t2 = time()
sys.stdout = sys.__stdout__ # restoring the stdout pipe to normal
print('Elapsed time ', t2 - t1)

print('Total elapsed time ',  t2 - t0)
 

