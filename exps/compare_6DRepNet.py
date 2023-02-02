
'''Although not using GPU, this version with ONNX weights running on CPU is super faster'''

__author__ = 'Huayi Zhou'

'''

Put this file under the main folder of codes project 6DRepNet

$ git clone https://github.com/thohemp/6DRepNet

usage:
python compare_6DRepNet.py --root-imgdir /path/to/root/imgdir \
    --json-file /path/to/prepared/json/file

e.g.:
python compare_6DRepNet.py --root-imgdir /datasdc/zhouhuayi/dataset/AGORA/HPE/images/validation --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/agora_m_1280_e300_t40_lw010/weights/val_best_c0.001_i0.65_pd_full.json --debug
[results]
Inference one image taking time: 0.048380501717018305
frontal face number: 3413; MAE_frontal: 25.4019, [pitch_error, yaw_error, roll_error]: 24.4108, 31.9957, 19.7991
face number: 6715; MAE: 42.1686, [pitch_error, yaw_error, roll_error]: 31.4113, 69.857, 25.2374
[results][2023-01-15][fix full-range bug]
Inference one image taking time: 0.038844961228028344
frontal face number: 3413; MAE_frontal: 25.5013, [pitch_error, yaw_error, roll_error]: 24.4108, 32.2938, 19.7991
face number: 6715; MAE: 42.0423, [pitch_error, yaw_error, roll_error]: 31.4113, 69.4782, 25.2374

python compare_6DRepNet.py --root-imgdir /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/images/val --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/cmu_m_1280_e200_t40_lw010/weights/val_best_c0.001_i0.65_pd_full.json --debug
[results]
Inference one image taking time: 0.03836491722721341
frontal face number: 15886; MAE_frontal: 15.1054, [pitch_error, yaw_error, roll_error]: 15.2149, 16.574, 13.5272
face number: 31976; MAE: 35.3256, [pitch_error, yaw_error, roll_error]: 23.1612, 62.547, 20.2686
[results][2023-01-15][fix full-range bug]
Inference one image taking time: 0.029975849789879674
frontal face number: 15886; MAE_frontal: 15.1304, [pitch_error, yaw_error, roll_error]: 15.2149, 16.649, 13.5272
face number: 31976; MAE: 35.2724, [pitch_error, yaw_error, roll_error]: 23.1612, 62.3874, 20.2686

'''

import os

import argparse
import time
import json
import cv2
import torch

from PIL import Image
from tqdm import tqdm
import numpy as np
from math import cos, sin, pi

from model import SixDRepNet
from torchvision import transforms
from utils import compute_euler_angles_from_rotation_matrices

# os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def crop_image(img_path, bbox, scale=1.0):
    img = cv2.imread(img_path)
    img_h, img_w, img_c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x_min, y_min, x_max, y_max = bbox
    # img_rgb = img_rgb[y_min:y_max, x_min:x_max]
    
    # enlarge the head/face bounding box
    # scale_ratio = 1.25
    scale_ratio = scale
    center_x, center_y = (x_max + x_min)/2, (y_max + y_min)/2
    face_w, face_h = x_max - x_min, y_max - y_min
    # new_w, new_h = face_w*scale_ratio, face_h*scale_ratio
    new_w = max(face_w*scale_ratio, face_h*scale_ratio)
    new_h = max(face_w*scale_ratio, face_h*scale_ratio)
    new_x_min = max(0, int(center_x-new_w/2))
    new_y_min = max(0, int(center_y-new_h/2))
    new_x_max = min(img_w-1, int(center_x+new_w/2))
    new_y_max = min(img_h-1, int(center_y+new_h/2))
    img_rgb = img_rgb[new_y_min:new_y_max, new_x_min:new_x_max]
    
    left = max(0, -int(center_x-new_w/2))
    top = max(0, -int(center_y-new_h/2))
    right = max(0, int(center_x+new_w/2) - img_w + 1)
    bottom = max(0, int(center_y+new_h/2) - img_h + 1)
    img_rgb = cv2.copyMakeBorder(img_rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    return img_rgb


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
   
    
    gpu_id = 3
    img_H = 256
    img_W = 256
    snapshot_path = "weights/6DRepNet_300W_LP_AFLW2000.pth"
    
    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(
                                              224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])])

    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='',
                        deploy=True,
                        pretrained=False,
                        gpu_id=gpu_id)
                        
    saved_state_dict = torch.load(snapshot_path, map_location='cpu')
    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)    
    model.cuda(gpu_id)
    model.eval()


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
        
        croped_frame = crop_image(img_path, bbox, scale=1.0)
        croped_resized_frame = cv2.resize(croped_frame, (img_W, img_H))  # h,w -> 256,256
        
        img_rgb = croped_resized_frame[..., ::-1]  # bgr --> rgb
        PIL_image = Image.fromarray(img_rgb)  # numpy array --> PIL image
        img_input = transformations(PIL_image)
        img_input = torch.Tensor(img_input).cuda(gpu_id)
        
        R_pred = model(img_input.unsqueeze(0))  # hwc --> nhwc
        euler = compute_euler_angles_from_rotation_matrices(R_pred, full_range=True)*180/np.pi
        p_pred_deg = euler[:, 0].cpu().detach().numpy()
        y_pred_deg = euler[:, 1].cpu().detach().numpy()
        r_pred_deg = euler[:, 2].cpu().detach().numpy()
        
        yaw, pitch, roll = y_pred_deg[0], p_pred_deg[0], r_pred_deg[0]
        
        if yaw > 360: yaw = yaw - 360
        if yaw < -360: yaw = yaw + 360
        if pitch > 360: pitch = pitch - 360
        if pitch < -360: pitch = pitch + 360
        if roll > 360: roll = roll - 360
        if roll < -360: roll = roll + 360
        
        t2 = time.time()
        taking_time_list.append(t2-t1)
        
        # pitch = pd_results['pitch']
        # yaw = pd_results['yaw']
        # roll = pd_results['roll']
        
        
        pd_poses.append([pitch, yaw, roll])
        gt_poses.append([gt_pitch, gt_yaw, gt_roll])
        
        if abs(gt_yaw) < 90:
            pd_poses_frontal.append([pitch, yaw, roll])
            gt_poses_frontal.append([gt_pitch, gt_yaw, gt_roll])

        if args.debug:
            save_img_path = "./tmp_test/"+str(ind).zfill(0)+\
                "_p"+str(round(gt_pitch, 2))+"v"+str(round(pitch, 2))+\
                "_y"+str(round(gt_yaw, 2))+"v"+str(round(yaw, 2))+\
                "_r"+str(round(gt_roll, 2))+"v"+str(round(roll, 2))+".jpg"

            cv2.rectangle(img_ori, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)
            img_ori = draw_axis(img_ori, yaw, pitch, roll, 
                tdx=(bbox[0]+bbox[2])/2, tdy=(bbox[1]+bbox[3])/2, size=100)
            cv2.imwrite(save_img_path, img_ori)

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
    parser = argparse.ArgumentParser(description='6DRepNet inference pipeline')
    
    parser.add_argument('--root-imgdir', default='',
                        help='root path to multiple images')
    parser.add_argument('--json-file', default='',
                        help='json file path that contains multiple images and their head bboxes')
    parser.add_argument('--debug',  action='store_true', help='whether set into debug mode')
    
    args = parser.parse_args()
    main(args)