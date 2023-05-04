
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CALCULATING SUBGLACIAL HYDRAULIC GRADIENT, SINKS AND FLOW
%
% Author: Joscha Sommerkorn
% Last modified: 30.06.2020
%
% USER INPUT: This code requires the user to specify the time step, or
% year, that the calcualtions should be run for. 
%
% Note that this code uses local paths for importing and exporting data. 
% The hydraulic head raster files must therefore be in a folder called 
% "DTM100_head" in the same folder as this code. Make sure the 3 output 
% folders specified below are also present in the same folder as this code.
%
% The calculations utilize the MATLAB addon "TopoToolbox". The add-on needs
% to be installed before running the code. To install the add-on, go to
% the "HOME" tab in your MATLAB program and click on the "Add-Ons" button.
% In the emerging "Add-on explorer" window, search for "Topotoolbox", click
% on the first result, and click on the "Install" button. Then restart
% MATLAB and attempt to run this code. If the code still does not run due
% to problems with the Topotoolbox elements, check that the specified path 
% to the installed add-on is correct.
%
% For more information regarding the Topotoolbox add-on, visit: 
% https://topotoolbox.wordpress.com/
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

% Path to Topotoolbox add-on may be required, if so insert yours here:
addpath(genpath('C:\Users\thomajba\AppData\Roaming\MathWorks\MATLAB Add-Ons\Collections\TopoToolbox')) 

year = 11000; % DEFINE YEAR HERE!

%% Define loop

files = dir('\\astra.uio.no\astra-01\thomajba\joschatest_sinks7\DTM50_head\');
N = 15;

%% Start loop
for i = 1:N
    thisfile = files(i).name;
    disp('starting loop for iteration')
    disp (year)

    %% Importing

    filename_input = strcat('DTM50_head\',string(year),'.tif');
    HEAD = GRIDobj(filename_input); % import hydraulic head  raster

    disp('import complete')

    %% Calculations

    % calculate and show sinks
    DEMfilled = fillsinks(HEAD); % identify and fill sinks in hydraulic head raster
    sinks = DEMfilled-HEAD; % subtract the filled hydraulic head raster from original hydraulic head raster to isolate sinks
    sinks.Z(sinks.Z==0) = nan; % convert areas of no sinks to NaNs, making them invisible in output

    disp('sink calculations complete')

    % hydraulic gradient calculation from DEM with sinks filled
    grad = GRIDobj(HEAD); % make new empty GRIDobj
    [GX,GY] = gradient(DEMfilled.Z,50); % calculate gradient in each direction, positive gradient is south.
    grad.Z = sqrt( GX.^2 + GY.^2 ); % calculate directionless gradient and add to the new GRIDobj

    disp('gradient calculations complete')

    FD  = FLOWobj(HEAD); % flow routing
    FA = flowacc(FD); % flow accumulation

    disp('flow calculations complete')

    %% Exporting

    sink_out = strcat('DTM50_sinks\sink_DTM50_year',string(year));
    GRIDobj2geotiff(sinks,sink_out);

    grad_out = strcat('DTM50_gradient\grad_DTM50_year',string(year));
    GRIDobj2geotiff(grad,grad_out);

    flowacc_out = strcat('DTM50_flow\flow_DTM50_year',string(year));
    GRIDobj2geotiff(FA,flowacc_out);

    disp('export complete')
    
    year = year + 1000
end
