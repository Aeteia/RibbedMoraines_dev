#! /bin/bash

for file in ./14-1819_resamp/*.tif
do
saga_cmd ta_hydrology 15 -DEM $file -TWI ${file/_10m.tif/_TWI.tif}
mv ./14-1819_resamp/*_TWI.tif ./14-1819_TWI/.
done
