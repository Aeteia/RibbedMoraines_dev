%% README
% This script imports two GeoTIFFs for Hydraulic Head and Basal
% Temperature, and masks out regions in which hydrology would not have
% existed under the modelled basal temperature conditions.
%
% This is done by:
% a) Importing the two GeoTiffs using Topotoolbox
% b) Multiplying the Z values by each other to create a masked Z value.
% c) Importing the masked value into the original Hydraulic head, and
% exporting as a new raster.
%
% Be sure to define the correct directory names in the start and final
% sections. In addition, the GeoTIFFs MUST take the same spatial reference,
% so thus must be produced using Joscha Sommerkorn's hydraulic head script,
% and the associated BasalTtoGeoTIFF script by Thomas Barnes.
% Year in this script is defined as below, with 11000 being the end of the
% LGM in Fennoscandia.


% Define starting parameters
clear all
year = 11000;

%% Define filepaths for GeoTIFFs
% Define hydraulic head location
dirname_hhead = '\\astra.uio.no\astra-01\thomajba\AltaHHead\DTM50_hhead\';

% Define basal temperature location
dirname_AbT = '\\astra.uio.no\astra-01\thomajba\BasalT_Masking\BinarisedBasalT_Alta\';

disp('GeoTIFF directories identified')

%% Attempt FOR loop

files = dir('\\astra.uio.no\astra-01\thomajba\AltaHHead\DTM50_hhead\');
N = 15;

disp('file length identified')

for i = 1:N
    thisfile = files(i).name;
    
    %% Read in Hydraulic Head GeoTIFF
    
    HHeadinput = strcat(dirname_hhead,string(year),'.tif');
    HHEAD = GRIDobj(HHeadinput); % Import hydraulic head raster
    
    disp('Hydraulic Head read for year:')
    disp(year)
    
    %% Read in Basal Temperature GeoTIFF
    
    AbTinput = strcat(dirname_AbT,'BBT_',string(year),'.tif');
    AbT = GRIDobj(AbTinput); % Import Basal temperature raster
    
    disp('Basal Temperature read for year:')
    disp(year)
    
    %% Masking process
%    HHEAD.Z(HHEAD.Z > 3.0e+10) = NaN; % Make all Nodata values NoData
    AbTmasked = AbT.Z .* HHEAD.Z; % Mask out cold base conditions
    HHEAD.Z = AbTmasked; % Replace HHEAD Z variable with Masked values
    
    disp('Basal Temperature masked out for year:')
    disp(year)
    
    %% Write new Hydraulic Head GeoTIFF using TopoToolbox functions
    
    MaskHeadOut = strcat('\\astra.uio.no\astra-01\thomajba\AltaHHead\DTM50_hhead_AbTmasked\',string(year));
    GRIDobj2geotiff(HHEAD,MaskHeadOut);
    
    disp('Masked GeoTiff written for year:')
    disp(year)
    
    %% Prepare for next iteration
    
    year = year + 1000;
end