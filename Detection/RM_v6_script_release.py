## Classification of Ribbed Moraines from DEM data & derivatives
# Place this file in a root directory, with a sub-directory named DEM containing:
# Sub-directory: "Filtered", "Lake Data", "ver_x-y", Initial DEM .tif file

## Import Libraries
import cv2
import csv
import rasterio
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import richdem as rd
import sys
from sklearn.datasets import load_iris
from skimage.measure import label, regionprops, regionprops_table
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib.colors import LightSource
import time
import pkg_resources
import os

## Loop
# for file.tif in files....


## Read in Raster DEM Data
rast = rasterio.open('DEM/Finse/InputData/11-12_resamp/33-111-120_10m.tif')
arr = rast.read(1)

## Read in SAGA TWI
wetind = rasterio.open('DEM/Finse/InputData/11-12_TWI/33-111-120_TWI.tif')
wIndex = wetind.read(1)

## Read in lake data
lakemaskrast = rasterio.open('DEM/Finse/InputData/Lakes/Lakes_111-120.tif')
lakearr = lakemaskrast.read(1)

## Read in Løsmasse data
lmass = rasterio.open('DEM/Finse/InputData/Løsmasse/Løsmasse_111-120.tif')
lmassarr = lmass.read(1)

## Generate Morphometrics from Raster Input (calculation-based)
# Define filtering kernel
def kernel_square(nPix):
    print("Averaging kernel of "+ str(nPix) + " by " + str(nPix))
    kernel = np.empty([nPix, nPix])
    kernel.fill(1)
    kernel /= kernel.sum()
    return kernel
def smooth(mat, kernel):
    r = cv2.filter2D(mat, -1, kernel)
    print("Smoothing complete...")
    return r

# Generate 400m filtered DEM
# Default size 40
med_400 = smooth(arr, kernel_square(34))

# Generate general slope layer
r400 = rd.array(med_400, no_data=np.nan)
genslope = rd.TerrainAttribute(r400, attrib='slope_riserun')

# Generate median filter
# Default size 40, 3
med_300 = smooth(arr, kernel_square(40))
hpass300 = arr-med_300
med_40 = smooth(hpass300, kernel_square(4))

# Low pass filter
LPass = med_300

# High pass filter
HPass = arr - med_40

# Median filter
MED = med_40

## RichDEM Morphometrics (tool-based)
rmed40 = rd.rdarray(MED, no_data=np.nan)
slope = rd.TerrainAttribute(rmed40, attrib='slope_riserun')
curv = rd.TerrainAttribute(rmed40, attrib='curvature')

## Define clustering dataframe
df = pd.DataFrame()
df['filt_elev'] = MED.flatten()
df['genslope'] = genslope.flatten()
df['slope'] = slope.flatten()
df['curvature'] = curv.flatten()
df['TWI'] = wIndex.flatten()
df['pix_id'] = df.index

## Masking features
def lapogaus(x, y, sigma):
    op = -1/(np.pi * sigma **4)*(1-((x**2 + y**2)/(2*sigma**2)))*np.exp(-((x**2 + y**2)/(2*sigma**2)))
    return op
def laplacian_of_gaussian(n, sigma):
    x = np.arange(-n,n)
    y = np.arange(-n,n)
    Xs, Ys = np.meshgrid(x,y)
    kernel = lapogaus(Xs, Ys, sigma)
    return kernel
def convolve_array(arr, kernel):
    res = cv2.filter2D(arr, -1, kernel)
    return res
kernel = laplacian_of_gaussian(71, 15)

# Generate Laplacian Curvature raster
lapcurv = cv2.filter2D(arr, -1, kernel)
lcval = 0.051

# Generate large-scale geomorphology filter
biglopass = smooth(arr, kernel_square(100)) # 100 = 1km, 200 = 2km)
lplap = cv2.filter2D(biglopass, -1, kernel)
lplapval = 0.012






