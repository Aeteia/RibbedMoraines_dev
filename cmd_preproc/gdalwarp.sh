#! /bin/bash

#set indir = /mn/vann/thomajba/10-10
#set outdir = /mn/vann/thoajba/10-10_resamp
#echo $indir

for file in ./1718-1820/*.tif
do
gdalwarp -tr 10 10 -r bilinear $file ${file/.tif/_10m.tif}
mv ./1718-1820/*_10m.tif ./1718-1820_resamp/.
done
