## Script to clip and burn polygon layers by rasters.

from osgeo import ogr, gdal
import subprocess
import os
import glob
from pathlib import Path

## Define Count & Colours
Count = 0
ylw = '\33[43m'
wht = '\033[0m'

# Define directories
Extent_dir = "DEM/Finse/InputData/11-12_resamp/"
Lakes_in_dir = "DEM/GlobalLakes/"
Løsmasse_in_dir = "DEM/løsmasse/Nationwide/"
Out_dir = "DEM/Finse/ConversionAutomation/"


## Start Loop
for filename in os.listdir(Extent_dir):
    if filename.endswith(".tif"):
        index_fn = filename[3:10]

        # Setup Extent Raster
        ExtentRaster = Extent_dir + "33-" + index_fn + "_10m.tif"

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
        Count = Count + 1

        # Complete Text
        print(ylw + "Clipping complete for Løsmasse tile: " + str(Count) + ". Continuing to rasterise..." + wht)

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
        print(ylw + "Løsmasse Raster creation complete for tile: " + str(Count) + ". Continuing with lakes..." + wht)

        ## LAKES
        # Make clip command with ogr2ogr - default to shapefile format
        cmd = 'ogr2ogr ' + OutVector_Lakes + ' ' + LakesPoly + ' -clipsrc ' + extent

        # Call the command
        subprocess.call(cmd, shell=True)

        # Complete Text
        print(ylw + "Clipping complete for Lake tile: " + str(Count) + ". Continuing to rasterise..." + wht)

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
        print(ylw + "Lake Raster creation complete for tile: " + str(Count) + ". Starting next iteration..." + wht)

# Complete Run
print(ylw + "Package processing complete on iteration: " + str(Count) + ". Cleaning Up..." + wht)

## Clean-up

for item in Path(Out_dir).glob("*.shp"):
    item.unlink()
for item in Path(Out_dir).glob("*.shx"):
    item.unlink()
for item in Path(Out_dir).glob("*.prj"):
    item.unlink()
for item in Path(Out_dir).glob("*.dbf"):
    item.unlink()

print(ylw + "Cleanup complete. Terminating Process." + wht)
