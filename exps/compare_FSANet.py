

__author__ = 'Huayi Zhou'

'''


Put this file under the main folder of codes project FSA-Net_pytorch
git clone https://github.com/omasaht/headpose-fsanet-pytorch FSA-Net_pytorch


usage:
python compare_FSANet.py --root-imgdir /path/to/root/imgdir \
    --json-file /path/to/prepared/json/file

e.g.:
python compare_FSANet.py --root-imgdir /datasdc/zhouhuayi/dataset/AGORA/HPE/images/validation --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/agora_m_1280_e300_t40_lw010/weights/val_best_c0.001_i0.65_pd_frontal.json --debug
[results]
Inference one image taking time: 0.013793717896777136
face number: 3403; MAE: 18.9809, [pitch_error, yaw_error, roll_error]: 18.9733, 21.6935, 16.2759


python compare_FSANet.py --root-imgdir /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/images/val --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/cmu_m_1280_e200_t40_lw010/weights/val_best_c0.001_i0.65_pd_frontal.json --debug
[results]
Inference one image taking time: 0.016242261164280028
face number: 15871; MAE: 15.6144, [pitch_error, yaw_error, roll_error]: 16.343, 17.515, 12.9852

'''

import os
import argparse
import time
import json
import cv2

import onnxruntime

import numpy as np
from tqdm import tqdm
from pathlib import Path
from math import cos, sin, pi

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
    
    sess1 = onnxruntime.InferenceSession(f'{root_path}/FSA-Net_pytorch/pretrained/fsanet-1x1-iter-688590.onnx')
    sess2 = onnxruntime.InferenceSession(f'{root_path}/FSA-Net_pytorch/pretrained/fsanet-var-iter-688590.onnx')
    
    # sess1 + sess2  --> 2.37 MB
    
    
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
        face_roi = cv2.resize(face_roi,(64,64))
        face_roi = face_roi.transpose((2,0,1))
        face_roi = np.expand_dims(face_roi,axis=0)
        face_roi = (face_roi-127.5)/128
        face_roi = face_roi.astype(np.float32)

        # get headpose
        res1 = sess1.run(["output"], {"input": face_roi})[0]
        res2 = sess2.run(["output"], {"input": face_roi})[0]

        yaw, pitch, roll = np.mean(np.vstack((res1,res2)),axis=0)
        t2 = time.time()
        taking_time_list.append(t2-t1)

        pd_poses.append([pitch, yaw, roll])
        gt_poses.append([gt_pitch, gt_yaw, gt_roll])

        if args.debug:
            save_img_path = "./tmp/"+str(ind).zfill(0)+\
                "_p"+str(round(gt_pitch, 2))+"v"+str(round(pitch, 2))+\
                "_y"+str(round(gt_yaw, 2))+"v"+str(round(yaw, 2))+\
                "_r"+str(round(gt_roll, 2))+"v"+str(round(roll, 2))+".jpg"

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