#!/bin/bash

#This list is named as "Panoptic Studio DB Ver 1.2"

curPath=$(dirname "$0")
hdVideoNum=31

#Dance sequences
$curPath/getData_hdVideo.sh 170307_dance5 $hdVideoNum

#Toddler sequences
$curPath/getData_hdVideo.sh 160906_ian1 $hdVideoNum

#Others sequences
$curPath/getData_hdVideo.sh 170915_office1 $hdVideoNum
$curPath/getData_hdVideo.sh 160906_pizza1 $hdVideoNum
