
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CALCULATING HYDRAULIC HEAD
%
% Author: Joscha Sommerkorn
% Last modified: 29.06.2020
%
% USER INPUT: This code requires the user to specify the time step, or
% year, that the calculations should be run for. The file paths should also
% be checked by the user before running this code.
%
% IMPORTANT NOTES: 
% - This code uses global file paths. The code will therefore only work
% when the relevant file paths have been changed by the user.
%
% - This code reguires the MATLAB function file "deg2utm33N.m" which must 
% be situated in the same folder as this code.
%
% - The code works for most input DEM files, as long as these are in the 
% form of GEOtiff files with a uniform, square grid. Note that the 
% resolution of the input DEM defines the resolution of the output data.
%
% - For an overview of the steps of this code: see the flow diagram in the 
% method part of my master thesis (figure 4, page 26).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Minor modifications to the script made by Thomas Barnes
% Modifications allow for looping through a file system.

clear all
close all
clc

year = 1100; % DEFINE YEAR HERE

%% Importing DEM

filename_dem = '\\kant.uio.no\geo-geohyd-u1\thomajba\PhD\Patton_QGIS_Prelim\MosaicDEM\Alta50_Merged.tif';
% CHECK ABOVE FILE PATH - should lead to "DEMs\DTM100-old-soer-norge" of this data archive to reproduce master thesis results.

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

disp('Step 1/5 complete - import of DEM')

%% Importing and pre-processing of ice sheet data

filedir_ice = '\\kant.uio.no\geo-geohyd-u1\thomajba\PhD\patton_model_grids_for_study_regions\Barnes_eemian5_WGS\';
% CHECK ABOVE FILE PATH - should lead to "Ice_Data\NGRIP_N79b_HZxyz\" of this data archive to reproduce master thesis results.

% import and process initial topography
h_ini = csvread(strcat(filedir_ice,'Alta_Z3CInitial_12285_WGS.csv')); % import initial basal topography ice sheet data csv-files
[h_ini_X, h_ini_Y, h_ini_UTM]= deg2utm33N(h_ini(:,2), h_ini(:,1)); % change coordinates form decimal degrees to UTM33N

%% Attempt FOR loop

files = dir('\\kant.uio.no\geo-geohyd-u1\thomajba\PhD\patton_model_grids_for_study_regions\Barnes_eemian5_WGS\Alta_H3C__*_WGS.csv');
N = 15;

disp('file length identified')

for i = 1:N
    thisfile = files(i).name;
    disp('starting loop for iteration')
    disp (year)
    % import correct ice sheet data csv-files (t_ice = ice thickess; h_topo = subglacial topography) 
    if year<1000
        t_ice = csvread(strcat(filedir_ice,'Alta_H3C__',string(year),'_WGS.csv'));
        h_topo = csvread(strcat(filedir_ice,'Alta_Z3C__',string(year),'_WGS.csv'));
    else
        t_ice = csvread(strcat(filedir_ice,'Alta_H3C__',string(year),'_WGS.csv'));
        h_topo = csvread(strcat(filedir_ice,'Alta_Z3C__',string(year),'_WGS.csv'));
    end

    t_ice = t_ice(:,3); % extract correct column
    h_topo = h_topo(:,3); % extract correct column

    t_ice(t_ice == -9999) = NaN; % set -9999 to NaN
    h_topo(h_topo == -9999) = NaN; % set -9999 to NaN

    % calculate new grids from the imported data
    h_ice = t_ice + h_topo; % ice sheet surface elevation = ice thickness + subglacial topography
    h_iso = h_ini(:,3) - h_topo; % isostatic depression = initial topography - subglacial topography

    disp('Step 2/5 complete - import and pre-processing of ice sheet data')

    %% Interpolation of ice sheet data

    % interpolate isostatic depression data to DEM grid resolution and orientation
    F_iso = scatteredInterpolant(h_ini_X,h_ini_Y,h_iso); % interpolation
    h_iso_grid = F_iso({hx,hy}); % extract interpolated values for extent of input DEM
    h_iso_grid = rot90(h_iso_grid); % rotate grid to correct orientation

    % correct the input DEM for the calculated isostatic depression under the ice sheet.
    h_topo_corrected = h_dem - h_iso_grid; % subglacial topography = input DEM - isostatic depression

    disp('Step 3/5 complete - interpolation of isostatic depression data')

    % interpolate ice sheet surface elevation data to the DEM grid resolution and orientation
    F_ice = scatteredInterpolant(h_ini_X,h_ini_Y,h_ice); % interpolation
    h_ice_grid = F_ice({hx,hy}); % extract interpolated values for extent of input DEM
    h_ice_grid = rot90(h_ice_grid); % rotate grid to correct orientation

    disp('Step 4/5 complete - interpolation of ice sheet surface elevation data')

    %% Hydraulic potential calculation

    % Define physical input parameters:
    rho_w = 1000; % [kg/m^3]
    rho_i = 917; % [kg/m^3]
    g = 9.81; % [m/s^2]
    Fl = 0.925; % flotation factor (see Shackleton et. al., 2018)

    % calcuate hydraulic head, output in meter water equivalants (m.w.e.)
    head = h_topo_corrected + Fl .* (rho_i / rho_w) .* (h_ice_grid - h_topo_corrected);

    %% Exporting
    % creating new geotiff files of the model outputs - CHECK FILE PATHS!

    info = geotiffinfo(filename_dem); % Extract raster grid info from input DEM

    folder_head = '\\astra.uio.no\astra-01\thomajba\AltaHHead\DTM50_hhead\'; % CHECK FILE PATH
    filename_head = strcat(folder_head, string(year),'0');
    geotiffwrite(filename_head,head,R,'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag)

    folder_height = '\\astra.uio.no\astra-01\thomajba\AltaHHead\DTM50_height\'; % CHECK FILE PATH
    filename_height = strcat(folder_height, string(year),'0');
    geotiffwrite(filename_height,h_ice_grid,R,'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag)

    folder_iso = '\\astra.uio.no\astra-01\thomajba\AltaHHead\DTM50_isostasy\'; % CHECK FILE PATH
    filename_iso = strcat(folder_iso, string(year),'0');
    geotiffwrite(filename_iso,h_iso_grid,R,'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag)

    disp('Step 5/5 complete - exporting hydraulic head raster')
    
    year = year + 100
end


