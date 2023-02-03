#!/bin/bash

# This script downloads videos for a specific sequence:
# ./getData.sh [sequenceName] [numHDViews]
#
# e.g., to download 10 VGA camera views for the "sampleData" sequence:
# ./getData.sh sampleData 10 0
# 

datasetName=${1-sampleData}
numHDViews=${2-31} #Specify the number of hd views you want to donwload. Up to 31

# Select wget or curl, with appropriate options
if command -v wget >/dev/null 2>&1; then 
	WGET="wget -c"
	mO="-O"
elif command -v curl >/dev/null 2>&1; then
	WGET="curl -C -" 
	mO="-o"
else
	echo "This script requires wget or curl to download files."
	echo "Aborting."
	exit 1;
fi

# Each sequence gets its own subdirectory
mkdir $datasetName		
cd $datasetName


# Download calibration data
$WGET $mO calibration_${datasetName}.json http://domedb.perception.cs.cmu.edu/webdata/dataset/$datasetName/calibration_${datasetName}.json || rm -v calibration_${datasetName}.json


# 3D Face 
if [ ! -f hdFace3d.tar ]; then
$WGET $mO hdFace3d.tar http://domedb.perception.cs.cmu.edu/webdata/dataset/$datasetName/hdFace3d.tar || rm -v hdFace3d.tar 
fi


# Extract 3D Keypoints
if [ -f hdFace3d.tar ]; then
	tar -xf hdFace3d.tar
fi



#####################
# Download hd videos
#####################
mkdir -p hdVideos
panel=0
nodes=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
for (( c=0; c<$numHDViews; c++))
do
  fileName=$(printf "hdVideos/hd_%02d_%02d.mp4" ${panel} ${nodes[c]})
  echo $fileName;
  #Download and delete if the file is blank
	cmd=$(printf "$WGET $mO hdVideos/hd_%02d_%02d.mp4 http://domedb.perception.cs.cmu.edu/webdata/dataset/$datasetName/videos/hd_shared_crf20/hd_%02d_%02d.mp4 || rm -v $fileName" ${panel} ${nodes[c]} ${panel} ${nodes[c]})
	eval $cmd
done

