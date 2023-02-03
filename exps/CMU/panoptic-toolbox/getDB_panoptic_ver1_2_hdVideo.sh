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

#Musical Instruments
$curPath/getData_hdVideo.sh 171026_cello3 $hdVideoNum
$curPath/getData_hdVideo.sh 161029_piano4 $hdVideoNum

#SocialGame sequences
$curPath/getData_hdVideo.sh 160422_ultimatum1 $hdVideoNum
$curPath/getData_hdVideo.sh 160224_haggling1 $hdVideoNum

#Dance sequences
$curPath/getData_hdVideo.sh 170307_dance5 $hdVideoNum

#Toddler sequences
$curPath/getData_hdVideo.sh 160906_ian1 $hdVideoNum

#Others sequences
$curPath/getData_hdVideo.sh 170915_office1 $hdVideoNum
$curPath/getData_hdVideo.sh 160906_pizza1 $hdVideoNum


#*** 4 other more names list ***
#Social Games (Haggling)
$curPath/getData_hdVideo.sh 170221_haggling_b3 $hdVideoNum
$curPath/getData_hdVideo.sh 170224_haggling_a3 $hdVideoNum
$curPath/getData_hdVideo.sh 170228_haggling_b1 $hdVideoNum
$curPath/getData_hdVideo.sh 170407_haggling_a2 $hdVideoNum
