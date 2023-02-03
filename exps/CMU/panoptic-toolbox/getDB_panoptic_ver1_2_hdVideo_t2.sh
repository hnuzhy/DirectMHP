#!/bin/bash

#This list is named as "Panoptic Studio DB Ver 1.2"

curPath=$(dirname "$0")
hdVideoNum=31

#Musical Instruments
$curPath/getData_hdVideo.sh 171026_cello3 $hdVideoNum
$curPath/getData_hdVideo.sh 161029_piano4 $hdVideoNum

#SocialGame sequences
$curPath/getData_hdVideo.sh 160422_ultimatum1 $hdVideoNum
$curPath/getData_hdVideo.sh 160224_haggling1 $hdVideoNum
