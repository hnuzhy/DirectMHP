
'''Too slow inference speed'''

__author__ = 'Huayi Zhou'

'''

usage:
python compare_dad3dnet.py --root-imgdir /path/to/root/imgdir \
    --json-file /path/to/prepared/json/file

e.g.:
python compare_dad3dnet.py --root-imgdir /datasdc/zhouhuayi/dataset/AGORA/HPE/images/validation --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/agora_m_1280_e300_t40_lw010/weights/val_best_c0.001_i0.65_pd_full.json --debug
[results]
Inference one image taking time: 0.018059632885267938
frontal face number: 3741; MAE_frontal: 35.1075, [pitch_error, yaw_error, roll_error]: 41.2611, 22.4362, 41.6252
face number: 7414; MAE: 80.1786, [pitch_error, yaw_error, roll_error]: 85.2124, 68.098, 87.2253
[results][2023-01-14]
Inference one image taking time: 0.01842204154443272
frontal face number: 3413; MAE_frontal: 32.6388, [pitch_error, yaw_error, roll_error]: 38.889, 19.987, 39.0404
face number: 6715; MAE: 80.2083, [pitch_error, yaw_error, roll_error]: 86.3455, 65.9651, 88.3143

python compare_dad3dnet.py --root-imgdir /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/images/val --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/cmu_m_1280_e200_t40_lw010/weights/val_best_c0.001_i0.65_pd_full.json --debug
[results]
Inference one image taking time: 0.02023921753516816
frontal face number: 16396; MAE_frontal: 21.7698, [pitch_error, yaw_error, roll_error]: 26.1876, 11.4288, 27.6928
face number: 32604; MAE: 80.5461, [pitch_error, yaw_error, roll_error]: 88.7443, 58.7891, 94.1048
[results][2023-01-14]
Inference one image taking time: 0.018940799204607884
frontal face number: 15886; MAE_frontal: 18.9887, [pitch_error, yaw_error, roll_error]: 22.4626, 10.58, 23.9235
face number: 31976; MAE: 79.7676, [pitch_error, yaw_error, roll_error]: 87.6178, 58.6636, 93.0214

'''

import os
import argparse
import time
import json
import cv2

import numpy as np
from tqdm import tqdm
from pathlib import Path
from math import cos, sin, pi

from model_training.model.flame import calculate_rpy, FlameParams, FLAME_CONSTS
from pytorch_toolbelt.utils import read_rgb_image
from predictor import FaceMeshPredictor
predictor_dad3dnet = FaceMeshPredictor.dad_3dnet()


np.set_printoptions(suppress=True)

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

root_path = str(Path(__file__).absolute().parent.parent)


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)
    return img


def main(args):
 
    # with open(args.json_file, "r") as json_f:
        # gt_results_dict = json.load(json_f)
        
        
    with open(args.json_file, "r") as json_f:
        pd_results_list = json.load(json_f)
        
    pd_poses = []  # predicted pose collection
    gt_poses = []  # ground-truth pose collection
    pd_poses_frontal = []  # predicted pose collection of frontal face
    gt_poses_frontal = []  # ground-truth pose collection of frontal face
    taking_time_list = []  # how many ms per face
    
    for ind, pd_results in enumerate(tqdm(pd_results_list)):
        if args.debug and ind > 50: break  # for testing
        
        img_path = os.path.join(args.root_imgdir, str(pd_results["image_id"])+".jpg")
        img_ori = cv2.imread(img_path)
        
        # bbox = pd_results['bbox'] # bbox default format is [x0,y0,w,h], should be converted to [x0,y0,x1,y1]
        bbox = pd_results['gt_bbox']
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

        gt_pitch = pd_results['gt_pitch']
        gt_yaw = pd_results['gt_yaw']
        gt_roll = pd_results['gt_roll']
        
        t1 = time.time()
        [x1, y1, x2, y2] = [int(i) for i in bbox]
        face_roi = img_ori[y1:y2+1,x1:x2+1]
    
        cropped_img_path = "./temp_cropped_img.jpg"
        cv2.imwrite(cropped_img_path, face_roi)
        image = read_rgb_image(cropped_img_path)
        predictions = predictor_dad3dnet(image)
        params_3dmm = predictions["3dmm_params"].float()
        flame_params = FlameParams.from_3dmm(params_3dmm, FLAME_CONSTS)
        rpy = calculate_rpy(flame_params)
        yaw, pitch, roll = rpy.yaw, rpy.pitch, rpy.roll
        
        t2 = time.time()
        taking_time_list.append(t2-t1)

        pd_poses.append([pitch, yaw, roll])
        gt_poses.append([gt_pitch, gt_yaw, gt_roll])
        
        if abs(gt_yaw) < 90:
            pd_poses_frontal.append([pitch, yaw, roll])
            gt_poses_frontal.append([gt_pitch, gt_yaw, gt_roll])

        if args.debug:
            save_img_path = "./tmp/"+str(ind).zfill(2)+"#"+str(id).zfill(2)+\
                "_p"+str(round(gt_pitch, 2))+"#"+str(np.round(pitch, 2))+\
                "_y"+str(round(gt_yaw, 2))+"#"+str(np.round(yaw, 2))+\
                "_r"+str(round(gt_roll, 2))+"#"+str(np.round(roll, 2))+".jpg"

            img_ori_copy = cv2.rectangle(img_ori.copy(), (int(bbox[0]), int(bbox[1])), 
                (int(bbox[2]), int(bbox[3])), (255,255,255), 2)
            img_ori_copy = draw_axis(img_ori_copy, yaw, pitch, roll, 
                tdx=(bbox[0]+bbox[2])/2, tdy=(bbox[1]+bbox[3])/2, size=100)
            cv2.imwrite(save_img_path, img_ori_copy)

        ind += 1
    os.remove(cropped_img_path)
    
    
    '''print all results'''
    print("Inference one image taking time:", sum(taking_time_list[1:])/len(taking_time_list[1:]))

    error_list_frontal = np.abs(np.array(pd_poses_frontal) - np.array(gt_poses_frontal))
    error_list_frontal = np.min((error_list_frontal, 360 - error_list_frontal), axis=0)
    pose_matrix_frontal = np.mean(error_list_frontal, axis=0)
    MAE_frontal = np.mean(pose_matrix_frontal)
    print("frontal face number: %d; MAE_frontal: %s, [pitch_error, yaw_error, roll_error]: %s, %s, %s"%(
        len(error_list_frontal), round(MAE_frontal, 4), round(pose_matrix_frontal[0], 4),
        round(pose_matrix_frontal[1], 4), round(pose_matrix_frontal[2], 4)))

    error_list = np.abs(np.array(gt_poses) - np.array(pd_poses))
    error_list = np.min((error_list, 360 - error_list), axis=0)
    pose_matrix = np.mean(error_list, axis=0)
    MAE = np.mean(pose_matrix)
    print("face number: %d; MAE: %s, [pitch_error, yaw_error, roll_error]: %s, %s, %s"%(len(taking_time_list),
        round(MAE, 4), round(pose_matrix[0], 4), round(pose_matrix[1], 4), round(pose_matrix[2], 4)))
        
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FAN inference pipeline')
    
    parser.add_argument('--root-imgdir', default='',
                        help='root path to multiple images')
    parser.add_argument('--json-file', default='',
                        help='json file path that contains multiple images and their head bboxes')
    parser.add_argument('--debug',  action='store_true', help='whether set into debug mode')
    
    args = parser.parse_args()
    main(args)