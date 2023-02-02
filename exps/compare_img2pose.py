
'''Too slow inference speed'''

__author__ = 'Huayi Zhou'

'''

git clone https://github.com/vitoralbiero/img2pose ./img2pose

Put this file under the main folder of codes project img2pose

usage:
python compare_img2pose.py --root-imgdir /path/to/root/imgdir \
    --json-file /path/to/prepared/json/file


e.g.:
python compare_img2pose.py --root-imgdir /datasdc/zhouhuayi/dataset/AGORA/HPE/images/validation --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/agora_m_1280_e300_t40_lw010/weights/val_best_c0.001_i0.65_pd_frontal.json --debug
[results]
Inference one image taking time: 0.019194672508227584
face number: 3138; MAE: 19.9507, [pitch_error, yaw_error, roll_error]: 22.1878, 17.2238, 20.4407


python compare_img2pose.py --root-imgdir /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/images/val --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/cmu_m_1280_e200_t40_lw010/weights/val_best_c0.001_i0.65_pd_frontal.json --debug
[results]
Inference one image taking time: 0.019501390085946375
face number: 15724; MAE: 15.0667, [pitch_error, yaw_error, roll_error]: 16.6038, 13.0171, 15.5792

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


import torch
from torchvision import transforms
from PIL import Image
from scipy.spatial.transform import Rotation
from img2pose import img2poseModel
from model_loader import load_model


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


def convert_to_aflw(rotvec, is_rotvec=True):
    if is_rotvec:
        rotvec = Rotation.from_rotvec(rotvec).as_matrix()
    rot_mat_2 = np.transpose(rotvec)
    angle = Rotation.from_matrix(rot_mat_2).as_euler('xyz', degrees=True)

    return np.array([angle[0], -angle[1], -angle[2]])  # Pitch, Yaw, Roll

    
def main(args):
 
    transform = transforms.Compose([transforms.ToTensor()])

    DEPTH = 18
    MAX_SIZE = 1400
    MIN_SIZE = 400

    POSE_MEAN = "./models/WIDER_train_pose_mean_v1.npy"
    POSE_STDDEV = "./models/WIDER_train_pose_stddev_v1.npy"
    # MODEL_PATH = "./models/img2pose_v1_ft_300w_lp.pth"
    MODEL_PATH = "./models/img2pose_v1.pth"  # 161 MB

    threed_points = np.load('./pose_references/reference_3d_68_points_trans.npy')

    pose_mean = np.load(POSE_MEAN)
    pose_stddev = np.load(POSE_STDDEV)

    img2pose_model = img2poseModel(
        DEPTH, MIN_SIZE, MAX_SIZE, 
        pose_mean=pose_mean, pose_stddev=pose_stddev,
        threed_68_points=threed_points,
        rpn_pre_nms_top_n_test=500,
        rpn_post_nms_top_n_test=10,
    )
    load_model(img2pose_model.fpn_model, MODEL_PATH, cpu_mode=str(img2pose_model.device) == "cpu", model_only=True)
    img2pose_model.evaluate()
    
    
    total_failures = 0
    
    
    with open(args.json_file, "r") as json_f:
        pd_results_list = json.load(json_f)

    pd_poses = []  # predicted pose collection
    gt_poses = []  # ground-truth pose collection
    taking_time_list = []  # how many ms per face

    for ind, pd_results in enumerate(tqdm(pd_results_list)):
        if args.debug and ind > 50: break  # for testing
        
        img_path = os.path.join(args.root_imgdir, str(pd_results["image_id"])+".jpg")
        img_ori = cv2.imread(img_path)
        
        bbox = pd_results['bbox'] # bbox default format is [x0,y0,w,h], should be converted to [x0,y0,x1,y1]
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

        gt_pitch = pd_results['gt_pitch']
        gt_yaw = pd_results['gt_yaw']
        gt_roll = pd_results['gt_roll']
        
        
        t1 = time.time()
        [x1, y1, x2, y2] = [int(i) for i in bbox]
        face_roi = img_ori[y1:y2+1,x1:x2+1]
        
        # preprocess headpose model input
        face_roi = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))  # opencv --> PIL

        # get headpose
        res = img2pose_model.predict([transform(face_roi)])
        
        res = res[0]
        bboxes = res["boxes"].cpu().numpy().astype('float')

        if len(bboxes) == 0:
            total_failures += 1
            continue
            
        max_score = 0
        best_index = -1
        for i in range(len(bboxes)):
            score = res["scores"][i]
            if score > max_score:
                max_score = score
                best_index = i  
         
        pose_pred = res["dofs"].cpu().numpy()[best_index].astype('float')
        pose_pred = np.asarray(pose_pred.squeeze())
        pose_pred[:3] = convert_to_aflw(pose_pred[:3])
        
        [pitch, yaw, roll] = pose_pred[:3]
        
        t2 = time.time()
        taking_time_list.append(t2-t1)

        pd_poses.append([pitch, yaw, roll])
        gt_poses.append([gt_pitch, gt_yaw, gt_roll])

        if args.debug:
            save_img_path = "./tmp/"+str(ind).zfill(0)+\
                "_p"+str(round(gt_pitch, 2))+"v"+str(np.round(pitch, 2))+\
                "_y"+str(round(gt_yaw, 2))+"v"+str(np.round(yaw, 2))+\
                "_r"+str(round(gt_roll, 2))+"v"+str(np.round(roll, 2))+".jpg"

            cv2.rectangle(img_ori, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)
            img_ori = draw_axis(img_ori, yaw, pitch, roll, 
                tdx=(bbox[0]+bbox[2])/2, tdy=(bbox[1]+bbox[3])/2, size=100)
            cv2.imwrite(save_img_path, img_ori)

    '''print all results'''
    print("Inference one image taking time:", sum(taking_time_list[1:])/len(taking_time_list[1:]))
    
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