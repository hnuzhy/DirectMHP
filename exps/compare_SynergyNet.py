

__author__ = 'Huayi Zhou'

'''


Put this file under the main folder of codes project SynergyNet
git clone https://github.com/choyingw/SynergyNet SynergyNet
https://drive.google.com/file/d/1SQsMhvAmpD1O8Hm0yEGom0C0rXtA0qs8/view [3dmm_data] data link
https://drive.google.com/file/d/1BVHbiLTfX6iTeJcNbh-jgHjWDoemfrzG/view [pretrained weight] data link

usage:
python compare_SynergyNet.py --root-imgdir /path/to/root/imgdir \
    --json-file /path/to/prepared/json/file

e.g.:
python compare_SynergyNet.py --root-imgdir /datasdc/zhouhuayi/dataset/AGORA/HPE/images/validation --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/agora_m_1280_e300_t40_lw010/weights/val_best_c0.001_i0.65_pd_frontal.json --debug
[results][2023-01-15]
Inference one image taking time: 0.007218158028596455
face number: 3415; MAE: 42.212, [pitch_error, yaw_error, roll_error]: 35.5837, 39.5468, 51.5054

python compare_SynergyNet.py --root-imgdir /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/images/val --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/cmu_m_1280_e200_t40_lw010/weights/val_best_c0.001_i0.65_pd_frontal.json --debug
[results][2023-01-15]
Inference one image taking time: 0.006917836413339054
face number: 15885; MAE: 24.6768, [pitch_error, yaw_error, roll_error]: 23.518, 27.5607, 22.9518
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

import types
import torch
import torchvision.transforms as transforms
from utils.ddfa import ToTensor, Normalize
from utils.inference import predict_pose, predict_sparseVert, predict_denseVert, crop_img
from model_building import SynergyNet


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

    IMG_SIZE = 120  # Following 3DDFA-V2, we also use 120x120 resolution
    transform = transforms.Compose([ToTensor(), Normalize(mean=127.5, std=128)])

    # load pre-tained model
    checkpoint_fp = 'pretrained/best.pth.tar' 
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']

    args_SynergyNet = types.SimpleNamespace()
    args_SynergyNet.arch = 'mobilenet_v2'
    args_SynergyNet.img_size = 120
    args_SynergyNet.devices_id = [0]

    model = SynergyNet(args_SynergyNet)
    model_dict = model.state_dict()

    # because the model is trained by multiple gpus, prefix 'module' should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]

    model.load_state_dict(model_dict, strict=False)
    model = model.cuda()
    model.eval()
    

    with open(args.json_file, "r") as json_f:
        pd_results_list = json.load(json_f)

    pd_poses = []  # predicted pose collection
    gt_poses = []  # ground-truth pose collection
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
        # [x1, y1, x2, y2] = [int(i) for i in bbox]
        # face_roi = img_ori[y1:y2+1,x1:x2+1]

        HCenter = (bbox[1] + bbox[3])/2
        WCenter = (bbox[0] + bbox[2])/2
        side_len = bbox[3]-bbox[1]
        margin = side_len * 0.75 // 2  # a larger bbox will result a worse MAE
        bbox[0], bbox[1], bbox[2], bbox[3] = WCenter-margin, HCenter-margin, WCenter+margin, HCenter+margin
        face_roi = crop_img(img_ori, bbox)
        
        img = cv2.resize(face_roi, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        input = transform(img).unsqueeze(0)
        with torch.no_grad():
            input = input.cuda()
            param = model.forward_test(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
        # inferences
        # lmks = predict_sparseVert(param, bbox, transform=True)
        # vertices = predict_denseVert(param, bbox, transform=True)
        angles, translation = predict_pose(param, bbox)
        yaw, pitch, roll = angles[0], angles[1], angles[2]
        
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