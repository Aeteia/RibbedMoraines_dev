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
filedir_csv = '\\kant.uio.no\geo-geohyd-u1\thomajba\PhD\patton_model_grids_for_study_regions\Barnes_eemian5_WGS\Alta\';

%% Define raster to convert for spatial reference for geotiffwrite

filename_dem = '\\kant.uio.no\geo-geohyd-u1\thomajba\PhD\Patton_QGIS_Prelim\MosaicDEM\Alta50_Merged.tif';
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
h_ini = csvread(strcat(filedir_csv,'Alta_Z3CInitial_12285_WGS.csv')); % import initial basal topography ice sheet data csv-files
[h_ini_X, h_ini_Y, h_ini_UTM]= deg2utm33N(h_ini(:,2), h_ini(:,1)); % change coordinates form decimal degrees to UTM33N

disp('Imported DEM for conversion of coordinate reference')


%% Attempt FOR loop to loop through Basal T files in the directory

files = dir('\\kant.uio.no\geo-geohyd-u1\thomajba\PhD\patton_model_grids_for_study_regions\Barnes_eemian5_WGS\Alta\');
N = 15;

disp('file length identified')

for i = 1:N
    thisfile = files(i).name;
    disp(year)
    % import correct csv file (basal temp) format Place_AbT__year_WGS
    basalT = csvread(strcat(filedir_csv,'Alta_AbT__',string(year),'_WGS.csv'));
    
    % Define the column variable for basal T values
    basalT_tempcol = basalT(:,3);
    
    %% Interpolate spatial reference
    % Interpolate basalT data to DEM grid resolution and orientation
    % This converts Basal T from its original WGS grid to UTM33
    F_basalT = scatteredInterpolant(h_ini_X,h_ini_Y,basalT_tempcol); % interpolation
    h_basalT_grid = F_basalT({hx,hy}); % extract interpolated values for extent of input DEM
    h_basalT_grid = rot90(h_basalT_grid); % rotate grid to correct orientation
    
    disp('Interpolated BasalT grid for year:')
    disp(year)
    
    %% Binarise raster where values below zero are 0, and above zero are 1
%    h_basalT_grid(h_basalT_grid >= 1.6e+30) = NaN; % Set bad values to NoData
    h_basalT_grid(h_basalT_grid <= 0) = NaN; % Set less than or equal to zero as NaN
    h_basalT_grid(h_basalT_grid > 0) = 1; % Set greater than zero to 1
        
    % Return the basal T value to column 3 of the csv
    basalT(:,3) = basalT_tempcol;
    
    disp('Binarised BasalT for year:')
    disp(year)
    
    %% Export binarised raster
    info = geotiffinfo(filename_dem); % Extract raster grid info from DEM
    folder_bbt = '\\astra.uio.no\astra-01\thomajba\BasalT_Masking\BinarisedBasalT_Alta\';
    filename_bbt = strcat(folder_bbt,'BBT_', string(year),'0');
    geotiffwrite(filename_bbt,h_basalT_grid,R,'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag)
    
    disp('Exported binarised raster for year:')
    disp(year)
    
    year = year + 100;
end