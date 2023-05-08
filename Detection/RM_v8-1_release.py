## Combination script of clip and burn process, and clustering.

from osgeo import ogr, gdal
import subprocess
import os
from pathlib import Path
import cv2
import rasterio
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import richdem as rd
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
import pandas as pd
from matplotlib.colors import LightSource
import time
import shutil

## Definitions
# Define directories
Extent_dir = "DEM/Finse/InputData/11-12_resamp/"
Lakes_in_dir = "DEM/GlobalLakes/"
Løsmasse_in_dir = "DEM/løsmasse/Nationwide/"
Out_dir = "DEM/Finse/InputData/"

## Define Count & Colours
Count_Clip = 0
ylw = '\33[43m'
wht = '\033[0m'

########################################################################################################################
############################################## CLIP & BURN PROCESS #####################################################
########################################################################################################################


## Start Loop
for filename in os.listdir(Extent_dir):
    if filename.endswith(".tif"):
        index_fn = filename[3:10]

        # Setup Extent Raster
        ExtentRaster = Extent_dir + "33-" + index_fn + "_10m.tif"
        
        # Fill Holes in Raster
        filledDEM = gdal.FillNodata(srcfile = ExtentRaster, targetBand = 1, maskBand = None,
                                        maxSearchDist = 30, smoothingIterations = 0)
        print(filledDEM)
        ExtentRaster = filledDEM
                                        
        # Get raster extent
        src = gdal.Open(ExtentRaster)
        ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
        sizeX = src.RasterXSize * xres
        sizeY = src.RasterYSize * yres
        lrx = ulx + sizeX
        lry = uly + sizeY
        src = None

        # Format the extent coorindates
        extent = '{0} {1} {2} {3}'.format(ulx, lry, lrx, uly);

        ## Define paths
        # Løsmasse
        LmassPoly = Løsmasse_in_dir + "Løsmasse_Fixed.shp"
        OutVector_Lmass = Out_dir + "Løsmasse_" + index_fn + ".shp"
        LmassIn = OutVector_Lmass
        LmassOut = LmassIn[:-3] + 'tif'

        # Lakes
        LakesPoly = Lakes_in_dir + "Innsjo_Innsjo.shp"
        OutVector_Lakes = Out_dir + "Lakes_" + index_fn + ".shp"
        LakesIn = OutVector_Lakes
        LakesOut = LakesIn[:-3] + 'tif'

        ## LØSMASSE
        # Make clip command with ogr2ogr - default to shapefile format
        cmd = 'ogr2ogr ' + OutVector_Lmass + ' ' + LmassPoly + ' -clipsrc ' + extent

        # Call the command
        subprocess.call(cmd, shell=True)

        # Add to count
        Count_Clip = Count_Clip + 1

        # Complete Text
        print(ylw + "Clipping complete for Løsmasse tile: " + str(index_fn) + ". Continuing to rasterise..." + wht)

        ## Rasterise Clipped polygon
        # Define gdal formats
        gdalformat = 'GTiff'
        datatype = gdal.GDT_Byte
        burnVal = 1

        # Get projection information from the reference raster
        img = gdal.Open(ExtentRaster, gdal.GA_ReadOnly)

        # Open shapefile
        LM_shp = ogr.Open(LmassIn)
        LM_shp_lyr = LM_shp.GetLayer()

        # Rasterise
        print("Rasterising Shapefile...")
        LM_out = gdal.GetDriverByName(gdalformat).Create(LmassOut, img.RasterXSize, img.RasterYSize, 1, datatype,
                                                         options=['COMPRESS=DEFLATE'])
        LM_out.SetProjection(img.GetProjectionRef())
        LM_out.SetGeoTransform(img.GetGeoTransform())

        # Write data to band 1
        Band = LM_out.GetRasterBand(1)
        Band.SetNoDataValue(0)
        gdal.RasterizeLayer(LM_out, [1], LM_shp_lyr, burn_values=[burnVal], options=["ATTRIBUTE=jordart"])

        # Close datasets
        Band = None
        LM_out = None
        img = None
        LM_shp = None

        # Build image overviews
        subprocess.call("gdaladdo --config COMPRESS_OVERVIEW DEFLATE " + LmassOut + " 2 4 8 16 32 64", shell=True)
        print("Done.")

        # Complete Text
        print(ylw + "Løsmasse Raster creation complete for tile: " + str(index_fn) + ". Continuing with lakes..." + wht)

        ## LAKES
        # Make clip command with ogr2ogr - default to shapefile format
        cmd = 'ogr2ogr ' + OutVector_Lakes + ' ' + LakesPoly + ' -clipsrc ' + extent

        # Call the command
        subprocess.call(cmd, shell=True)

        # Complete Text
        print(ylw + "Clipping complete for Lake tile: " + str(index_fn) + ". Continuing to rasterise..." + wht)

        ## Rasterise Clipped polygon
        # Define gdal formats
        gdalformat = 'GTiff'
        datatype = gdal.GDT_Byte
        burnVal = 1

        # Get projection information from the reference raster
        img = gdal.Open(ExtentRaster, gdal.GA_ReadOnly)

        # Open shapefile
        LK_shp = ogr.Open(LakesIn)
        LK_shp_lyr = LK_shp.GetLayer()

        # Rasterise
        print("Rasterising Shapefile...")
        LK_out = gdal.GetDriverByName(gdalformat).Create(LakesOut, img.RasterXSize, img.RasterYSize, 1, datatype,
                                                         options=['COMPRESS=DEFLATE'])
        LK_out.SetProjection(img.GetProjectionRef())
        LK_out.SetGeoTransform(img.GetGeoTransform())

        # Write data to band 1
        Band = LK_out.GetRasterBand(1)
        Band.SetNoDataValue(0)
        gdal.RasterizeLayer(LK_out, [1], LK_shp_lyr, burn_values=[burnVal], options=["ATTRIBUTE=Lake"])

        # Close datasets
        Band = None
        LK_out = None
        img = None
        LK_shp = None

        # Build image overviews
        subprocess.call("gdaladdo --config COMPRESS_OVERVIEW DEFLATE " + LakesOut + " 2 4 8 16 32 64", shell=True)
        print("Done.")

        # Complete Text
        print(ylw + "Lake Raster creation complete for tile: " + str(index_fn) + ". Starting next iteration..." + wht)

# Complete Run
print(ylw + "Package processing complete on iteration: " + str(index_fn) + ". Cleaning Up..." + wht)

## Clean-up & file moving
Lakes_dir = 'DEM/Finse/InputData/Lakes/'
Lmass_dir = 'DEM/Finse/InputData/Løsmasse/'

# Path DOES NOT look into subdirs on its own volition.
for item in Path(Out_dir).glob("*.shp"):
    item.unlink()
for item in Path(Out_dir).glob("*.shx"):
    item.unlink()
for item in Path(Out_dir).glob("*.prj"):
    item.unlink()
for item in Path(Out_dir).glob("*.dbf"):
    item.unlink()
for item in Path(Out_dir).glob("Lakes*.tif"):
    itemstr = str(item)
    shutil.move(item, Lakes_dir + itemstr[20:])
for item in Path(Out_dir).glob("Løsmasse*.tif"):
    itemstr = str(item)
    shutil.move(item, Lmass_dir + itemstr[20:])

print(ylw + "Cleanup complete." + wht)
print(ylw + "Waiting 5 seconds before continuing..." + wht)
time.sleep(5)
print(ylw + "Continuing to clustering process..." + wht)

########################################################################################################################
############################################## K-MEANS CLUSTERING ######################################################
########################################################################################################################

## Start Iteration Count & Colours
Count = 0

## Define Input Directories
DEM_dir = Extent_dir
TWI_dir = 'DEM/Finse/InputData/11-12_TWI/'

## Loop
# for file.tif in files....
for filename in os.listdir(DEM_dir):
    if filename.endswith(".tif"):
        index_fn = filename[3:10]

        ## Read in Raster DEM Data
        rast = rasterio.open(DEM_dir + "33-" + index_fn + "_10m.tif")
        arr = rast.read(1)

        ## Read in SAGA TWI
        wetind = rasterio.open(TWI_dir + "33-" + index_fn + "_TWI.tif")
        wIndex = wetind.read(1)

        ## Read in lake data
        lakemaskrast = rasterio.open(Lakes_dir + "Lakes_" + index_fn + ".tif")
        lakearr = lakemaskrast.read(1)

        ## Read in Løsmasse data
        lmass = rasterio.open(Lmass_dir + "Løsmasse_" + index_fn + ".tif")
        lmassarr = lmass.read(1)

        ## Generate Morphometrics from Raster Input (calculation-based)
        # Define filtering kernel
        def kernel_square(nPix):
            print("Averaging kernel of " + str(nPix) + " by " + str(nPix))
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
        r400 = rd.rdarray(med_400, no_data=np.nan)
        genslope = rd.TerrainAttribute(r400, attrib='slope_riserun')

        # Generate median filter
        # Default size 40, 3
        med_300 = smooth(arr, kernel_square(40))
        hpass300 = arr - med_300
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
            op = -1 / (np.pi * sigma ** 4) * (1 - ((x ** 2 + y ** 2) / (2 * sigma ** 2))) * np.exp(
                -((x ** 2 + y ** 2) / (2 * sigma ** 2)))
            return op
        def laplacian_of_gaussian(n, sigma):
            x = np.arange(-n, n)
            y = np.arange(-n, n)
            Xs, Ys = np.meshgrid(x, y)
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
        biglopass = smooth(arr, kernel_square(140))  # 100 = 1km, 200 = 2km)
        lplap = cv2.filter2D(biglopass, -1, kernel)
        lplapval = 0.012

        # Define mask
        maskarea = lapcurv < -lcval
        maskarea = lplap < -lplapval
        maskarea[genslope > 0.692] = True
        maskarea[lakearr == 1] = True
        df['mask'] = maskarea.flatten()
        df['pix_id'] = df.index
        df_sub = df.loc[df['mask'] == False]

        # Display mask
        plt.figure(figsize=(15, 15))
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
            miniBKmeans = cluster.MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048, random_state=seed,
                                                  **kwargs).fit(X)
            print('---> Mini-Batch Kmean finished in {}s'.format(np.round(time.time() - start_time), 0))
            df_centers = pd.DataFrame(miniBKmeans.cluster_centers_, columns=col_names)
            df_param['cluster_labels'] = miniBKmeans.labels_
            return df_centers, miniBKmeans, df_param['cluster_labels']
        df_scaled, scaler = scale_df(df_sub)

        ## Determine optical cluster count (default to disabled)
        # distortions = []
        # inertias = []
        # Xa = pd.DataFrame(df_scaled)
        # data = Xa[['slope', 'curv', 'wIndex', 'mask']]
        # sse = {}
        # for k in range(1, 11):
        #    kmeans = KMeans(n_clusters=k, max_iter=300).fit(data)
        #    data['clusters'] = kmeans.labels_
        #    sse[k] = kmeans.inertia_

        ## plot elbow
        # plt.figure()
        # plt.plot(list(sse.keys()), list(sse.values()))
        # plt.xlabel("Cluster count")
        # plt.ylabel("SSE")
        # plt.show()

        # Run clustering process
        df_centers, miniBKmeans, df_sub['cluster_labels'] = minibatch_kmeans_clustering(df_scaled, n_clusters=4,
                                                                                        n_cores=4, seed=123)
        df['cluster_sub'] = np.nan
        df.cluster_sub.loc[df_sub.pix_id] = df_sub.cluster_labels

        # Display output
        plt.figure(figsize=(15, 15))
        ls = LightSource(azdeg=225, altdeg=45)
        shade = ls.hillshade(arr, vert_exag=1, dx=10, dy=10, fraction=1.0)
        plt.imshow(shade, cmap=plt.cm.gray)
        plt.imshow(np.reshape(df.cluster_sub.values, med_40.shape))
        plt.title('Subtracted_Clustered_Map')

        # Save and Export output
        sel = np.reshape(df.cluster_sub.values, med_40.shape)
        with rasterio.open('DEM/Finse/OutputData/33_' + index_fn + "_Cluster.tif",
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

        ## Completion
        # Iteration message
        Count = Count + 1
        print(ylw + "Iteration: " + str(index_fn) + " Completed, starting next iteration..." + wht)


# Completion message
print(ylw + "Package finished clustering on iteration: " + str(Count) + " For tile designation: " + str(index_fn) + ". Terminating script." + wht)
