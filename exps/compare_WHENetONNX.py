
'''Although not using GPU, this version with ONNX weights running on CPU is super faster'''

__author__ = 'Huayi Zhou'

'''

Put this file under the main folder of codes project WHENet

$ git clone https://github.com/PINTO0309/HeadPoseEstimation-WHENet-yolov4-onnx-openvino
$ wget https://github.com/PINTO0309/HeadPoseEstimation-WHENet-yolov4-onnx-openvino/releases/download/v1.0.2/saved_model_224x224.tar.gz
$ tar -zxvf saved_model_224x224.tar.gz && rm saved_model_224x224.tar.gz


usage:
python compare_WHENetONNX.py --root-imgdir /path/to/root/imgdir \
    --json-file /path/to/prepared/json/file

e.g.:
python compare_WHENetONNX.py --root-imgdir /datasdc/zhouhuayi/dataset/AGORA/HPE/images/validation --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/agora_m_1280_e300_t40_lw010/weights/val_best_c0.001_i0.65_pd_full.json --debug
[results]
Inference one image taking time: 0.11055187362314
frontal face number: 3403; MAE_frontal: 23.9802, [pitch_error, yaw_error, roll_error]: 20.9284, 35.194, 15.8181
face number: 6692; MAE: 25.1804, [pitch_error, yaw_error, roll_error]: 21.8992, 37.0444, 16.5975


python compare_WHENetONNX.py --root-imgdir /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/images/val --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/cmu_m_1280_e200_t40_lw010/weights/val_best_c0.001_i0.65_pd_full.json --debug
[results]
Inference one image taking time: 0.13425157879707464
frontal face number: 15871; MAE_frontal: 25.7083, [pitch_error, yaw_error, roll_error]: 22.7132, 37.8637, 16.548
face number: 31964; MAE: 21.4637, [pitch_error, yaw_error, roll_error]: 19.8868, 29.8437, 14.6605

'''

import os

import argparse
import time
import json
import cv2

from tqdm import tqdm
import numpy as np
from math import cos, sin, pi

import onnxruntime

idx_tensor_yaw = [np.array(idx, dtype=np.float32) for idx in range(120)]
idx_tensor = [np.array(idx, dtype=np.float32) for idx in range(66)]


os.environ["CUDA_VISIBLE_DEVICES"] = '3'


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

def softmax(x):
    x -= np.max(x,axis=1, keepdims=True)
    a = np.exp(x)
    b = np.sum(np.exp(x), axis=1, keepdims=True)
    return a/b


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
    
    whenet_H = 224
    whenet_W = 224
    
    # WHENet
    whenet_input_name = None
    whenet_output_names = None
    whenet_output_shapes = None
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    whenet = onnxruntime.InferenceSession(f'saved_model_{whenet_H}x{whenet_W}/model_float32.onnx')  # 16.5 MB
    whenet_input_name = whenet.get_inputs()[0].name
    whenet_output_names = [output.name for output in whenet.get_outputs()]
    whenet_output_shapes = [output.shape for output in whenet.get_outputs()]
    assert whenet_output_shapes[0] == [1, 120] # yaw
    assert whenet_output_shapes[1] == [1, 66]  # roll
    assert whenet_output_shapes[2] == [1, 66]  # pitch

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
        
        bbox = pd_results['bbox'] # bbox default format is [x0,y0,w,h], should be converted to [x0,y0,x1,y1]
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

        gt_pitch = pd_results['gt_pitch']
        gt_yaw = pd_results['gt_yaw']
        gt_roll = pd_results['gt_roll']
        
        
        t1 = time.time()
        
        croped_frame = crop_image(img_path, bbox)
        croped_resized_frame = cv2.resize(croped_frame, (whenet_W, whenet_H))  # h,w -> 224,224
        rgb = croped_resized_frame[..., ::-1]  # bgr --> rgb
        rgb = ((rgb / 255.0) - mean) / std  # Normalization
        chw = rgb.transpose(2, 0, 1)  # hwc --> chw
        nchw = np.asarray(chw[np.newaxis, :, :, :], dtype=np.float32)  # chw --> nchw
        
        yaw, roll, pitch = whenet.run(
            output_names = whenet_output_names,
            input_feed = {whenet_input_name: nchw}
        )
        
        yaw = np.sum(softmax(yaw) * idx_tensor_yaw, axis=1) * 3 - 180
        pitch = np.sum(softmax(pitch) * idx_tensor, axis=1) * 3 - 99
        roll = np.sum(softmax(roll) * idx_tensor, axis=1) * 3 - 99
        
        yaw, pitch, roll = np.squeeze([yaw, pitch, roll])
        
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
    parser = argparse.ArgumentParser(description='WHENet inference pipeline')
    
    parser.add_argument('--root-imgdir', default='',
                        help='root path to multiple images')
    parser.add_argument('--json-file', default='',
                        help='json file path that contains multiple images and their head bboxes')
    parser.add_argument('--debug',  action='store_true', help='whether set into debug mode')
    
    args = parser.parse_args()
    main(args)