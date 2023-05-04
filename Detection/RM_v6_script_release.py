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
med_400 = smooth(arr, kernel_square(40))

# Generate general slope layer
r400 = rd.array(med_400, no_data=np.nan)
genslope = rd.TerrainAttribute(r400, attrib='slope_riserun')

# Generate median filter
# Default size 40, 3
med_300 = smooth(arr, kernel_square(40))
hpass300 = arr-med_300
med_40 = smooth(hpass300, kernel_square(3))

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
biglopass = smooth(arr, kernel_square(140)) # 100 = 1km, 200 = 2km)
lplap = cv2.filter2D(biglopass, -1, kernel)
lplapval = 0.012

# Define mask
maskarea = lapcurv<-lcval
maskarea = lplap<-lplapval
maskarea[genslope>0.692]=True
maskarea[lakearr==1]=True
df['mask'] = maskarea.flatten()
df['pix_id'] = df.index
df_sub = df.loc[df['mask']==False]

# Display mask
plt.figure(figsize=(15,15))
ls = LightSource(azdeg=225, altdeg=45)
shade = ls.hillshade(arr, vert_exag=1, dx=10, dy=10, fraction=1.0)
plt.imshow(shade, cmap=plt.cm.gray)
plt.imshow(np.reshape(maskarea, med_40.shape))
plt.title('Mask')


## Clustering via K-Means
# Setup algorithm
def scale_df(df_param, scaler=StandardScaler()):
    print('---> Scaling data prior to clustering')
    df_scaled = pd.DataFrame(scaler.fit_transform(df_param.values),
                             columns=df_param.columns, index=df_param.index)
    return df_scaled, scaler
def minibatch_kmeans_clustering(df_param, n_clusters=100, n_cores=4, seed=None, **kwargs):
    X = df_param.to_numpy()
    col_names = df_param.columns
    print('---> Clustering with Mini-Batch K-Means in {} clusters'.format(n_clusters))
    start_time = time.time()
    miniBKmeans = cluster.MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048, random_state=seed, **kwargs).fit(X)
    print('---> Mini-Batch Kmean finished in {}s'.format(np.round(time.time()-start_time), 0))
    df_centers = pd.DataFrame(miniBKmeans.cluster_centers_, columns=col_names)
    df_param['cluster_labels'] = miniBKmeans.labels_
    return df_centers, miniBKmeans, df_param['cluster_labels']
df_scaled, scaler = scale_df(df_sub)

## Determine optical cluster count (default to disabled)
#distortions = []
#inertias = []
#Xa = pd.DataFrame(df_scaled)
#data = Xa[['slope', 'curv', 'wIndex', 'mask']]
#sse = {}
#for k in range(1, 11):
#    kmeans = KMeans(n_clusters=k, max_iter=300).fit(data)
#    data['clusters'] = kmeans.labels_
#    sse[k] = kmeans.inertia_

## plot elbow
#plt.figure()
#plt.plot(list(sse.keys()), list(sse.values()))
#plt.xlabel("Cluster count")
#plt.ylabel("SSE")
#plt.show()

# Run clustering process
df_centers, miniBKmeans, df_sub['cluster_labels'] = minibatch_kmeans_clustering(df_scaled, n_clusters=4, n_cores=4, seed=123)
df['cluster_sub'] = np.nan
df.cluster_sub.loc[df_sub.pix_id] = df_sub.cluster_labels

# Display output
plt.figure(figsize=(15,15))
ls = LightSource(azdeg=225, altdeg=45)
shade = ls.hillshade(arr, vert_exag=1, dx=10, dy=10, fraction=1.0)
plt.imshow(shade, cmap=plt.cm.gray)
plt.imshow(np.reshape(df.cluster_sub.values, med_40.shape))
plt.title('Subtracted_Clustered_Map')

# Save and Export output
sel = np.reshape(df.cluster_sub.values, med_40.shape)
with rasterio.open('DEM/Finse/OutputData/33-111-120_Cluster.tif',
    'w',
    driver='GTiff',
    height=sel.shape[0],
    width=sel.shape[1],
    count=1,
    dtype=sel.dtype,
    crs={'init': 'EPSG:3045'},
    transform=rast.transform,
) as dst:
    dst.write(np.reshape(df.cluster_sub.values, med_40.shape), 1)