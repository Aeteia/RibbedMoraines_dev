%% README
% This script makes use of deg2utm33N.m, and thus must be in the same
% folder as it.
% This script:
% a) Imports a DEM as as spatial reference (likely can be streamlined)
% b) Imports a .csv file, and binarises it based on a series of parameters,
% in this case values below 0 degrees celsius are defined as 0, and above
% as 1.
% c) Interpolates and converts the imported WGS .csv file into UTM 33, thus
% allowing for easy export.
% d) Exports the .csv file as a geoTIFF file, using the same spatial
% reference as the DEM imported at the start.
% e) Loops through all files within the given directory
%
% This needs to be run separately for each region, as the file system
% definition uses "year" as the radical, rather than spatial definition.

%% Define parameters
clear all

year = 1100;
filedir_csv = '\\kant.uio.no\geo-geohyd-u1\thomajba\PhD\patton_model_grids_for_study_regions\Barnes_eemian5_WGS\Rjukan\';

%% Define raster to convert for spatial reference for geotiffwrite

filename_dem = '\\kant.uio.no\geo-geohyd-u1\thomajba\PhD\Patton_QGIS_Prelim\MosaicDEM\Rjukan50_Merged.tif';
[h_dem,R] = geotiffread(filename_dem); % Import DEM
h_dem = double(h_dem); % change number format to "double", i.e. decimal numbers.

if R.CellExtentInWorldX == R.CellExtentInWorldY % if-test to check if DEM input raster is suitable for calculations.
    disp('Check complete: Input DEM has a uniform, square grid.')
else
    disp('ERROR: Code unable to run because input DEM does not have a uniform, square grid.')
end

% extract extent of DEM input raster in X (east-west) and Y (north-south) directions.
hx=R.XWorldLimits(1)+R.CellExtentInWorldX/2:R.CellExtentInWorldX:R.XWorldLimits(2)-R.CellExtentInWorldX/2;
hy=R.YWorldLimits(1)+R.CellExtentInWorldY/2:R.CellExtentInWorldY:R.YWorldLimits(2)-R.CellExtentInWorldY/2;

% import and process initial topography
h_ini = csvread(strcat(filedir_csv,'Rjukan_Z3CInitial_12285_WGS.csv')); % import initial basal topography ice sheet data csv-files
[h_ini_X, h_ini_Y, h_ini_UTM]= deg2utm33N(h_ini(:,2), h_ini(:,1)); % change coordinates form decimal degrees to UTM33N

disp('Imported DEM for conversion of coordinate reference')

%% Export binarised raster
info = geotiffinfo(filename_dem); % Extract raster grid info from DEM
folder_bbt = '\\astra.uio.no\astra-01\thomajba\BasalT_Masking\initialDEM\';
filename_bbt = strcat(folder_bbt,'InitialDEM', string(year),'0');
geotiffwrite(filename_bbt,h_dem,R,'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag)
    
disp('initialDEM')