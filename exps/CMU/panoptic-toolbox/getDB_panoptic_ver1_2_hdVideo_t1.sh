#!/bin/bash

#This list is named as "Panoptic Studio DB Ver 1.2"

curPath=$(dirname "$0")
hdVideoNum=31

#Range of motion sequences
$curPath/getData_hdVideo.sh 171204_pose3 $hdVideoNum
$curPath/getData_hdVideo.sh 171026_pose3 $hdVideoNum

#Download All Haggling Sequences without downloading videos
$curPath/getData_hdVideo.sh 170221_haggling_m3 $hdVideoNum
$curPath/getData_hdVideo.sh 170404_haggling_a1 $hdVideoNum
$curPath/getData_hdVideo.sh 170407_haggling_b2 $hdVideoNum
