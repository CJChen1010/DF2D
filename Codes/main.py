import os, re
import shutil, sys
from datetime import datetime
import yaml
import subprocess as sp

param_file = "params.yaml"
params = yaml.safe_load(open(param_file, "r"))    

""" Running image registration"""
align_shell = [sys.executable, "AlignerDriver.py", param_file]
print(align_shell)
commandOut = sp.run(align_shell, stderr = sp.PIPE, text = True)

if commandOut.stderr != '':
    print(commandOut.stderr)
    sys.exit(1)

""" Running background subtraction (optional)"""
if params['background_subtraction']:
    subt_shell = [sys.executable, "backgroundSubtraction.py", param_file]
    print(subt_shell)
    commandOut = sp.run(subt_shell, stderr = sp.PIPE, text = True)
 
print(commandOut.stderr)

if commandOut.returncode:
    print("Error in background subtraction:")
    sys.exit(1)
    
""" Running stitching"""
stitch_shell = [sys.executable, "StitchDriver.py", param_file]
print(stitch_shell)
commandOut = sp.run(stitch_shell, stderr = sp.PIPE, text = True)

print(commandOut.stderr)
if commandOut.returncode:
    print("Error in stitching")
    print(commandOut.stderr)
    sys.exit(1)

""" to StarFish format"""
tosff_shell = [sys.executable, "toStarfishFormat.py", param_file]
print(tosff_shell)
commandOut = sp.run(tosff_shell, stderr = sp.PIPE, text = True)

print(commandOut.stderr)

if commandOut.returncode:
    print("Error in toStarfishFormat:")
    print(commandOut.stderr)
    sys.exit(1)

# """ Run Starfish decoding"""
dc_shell = [sys.executable, "starfishDARTFISHpipeline.py", param_file]
print(dc_shell)
commandOut = sp.run(dc_shell, stderr = sp.PIPE, text = True)


print(commandOut.stderr)

if commandOut.returncode:
    print("Error in Starfish decoding:")
    print(commandOut.stderr)
    sys.exit(1)


