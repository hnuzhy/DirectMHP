import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import torch
import argparse
import yaml
import cv2
import math
from math import cos, sin
import os.path as osp
import numpy as np

from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load


def plot_3axis_Zaxis(img, yaw, pitch, roll, tdx=None, tdy=None, size=50., limited=True, thickness=2):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    
    if tdx != None and tdy != None:
        face_x = tdx
        face_y = tdy
    else:
        height, width = img.shape[:2]
        face_x = width / 2
        face_y = height / 2

    # X-Axis (pointing to right) drawn in red
    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    
    # Y-Axis (pointing to down) drawn in green
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Plot head oritation line in black
    # scale_ratio = 5
    scale_ratio = 2
    base_len = math.sqrt((face_x - x3)**2 + (face_y - y3)**2)
    if face_x == x3:
        endx = tdx
        if face_y < y3:
            if limited:
                endy = tdy + (y3 - face_y) * scale_ratio
            else:
                endy = img.shape[0]
        else:
            if limited:
                endy = tdy - (face_y - y3) * scale_ratio
            else:
                endy = 0
    elif face_x > x3:
        if limited:
            endx = tdx - (face_x - x3) * scale_ratio
            endy = tdy - (face_y - y3) * scale_ratio
        else:
            endx = 0
            endy = tdy - (face_y - y3) / (face_x - x3) * tdx
    else:
        if limited:
            endx = tdx + (x3 - face_x) * scale_ratio
            endy = tdy + (y3 - face_y) * scale_ratio
        else:
            endx = img.shape[1]
            endy = tdy - (face_y - y3) / (face_x - x3) * (tdx - endx)
    # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (0,0,0), 2)
    # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (255,255,0), 2)
    cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (0,255,255), thickness)

    # X-Axis pointing to right. drawn in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),thickness)
    # Y-Axis pointing to down. drawn in green    
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,255,0),thickness)
    # Z-Axis (out of the screen) drawn in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),thickness)

    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--img-path', default='test_imgs/100024.jpg', help='path to image or dir')
    parser.add_argument('--data', type=str, default='data/agora_coco.yaml')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--weights', default='yolov5m6.pt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--thickness', type=int, default=2, help='thickness of Euler angle lines')

    args = parser.parse_args()

    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    device = select_device(args.device, batch_size=1)
    print('Using device: {}'.format(device))

    model = attempt_load(args.weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    dataset = LoadImages(args.img_path, img_size=imgsz, stride=stride, auto=True)
    dataset_iter = iter(dataset)
    
    for index in range(len(dataset)):
        
        (single_path, img, im0, _) = next(dataset_iter)
        
        if '_res' in single_path: continue
        
        print(index, single_path, "\n")
        
        img = torch.from_numpy(img).to(device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        out_ori = model(img, augment=True, scales=args.scales)[0]
        out = non_max_suppression(out_ori, args.conf_thres, args.iou_thres, num_angles=data['num_angles'])
        
        # predictions (Array[N, 9]), x1, y1, x2, y2, conf, class, pitch, yaw, roll
        bboxes = scale_coords(img.shape[2:], out[0][:, :4], im0.shape[:2]).cpu().numpy()  # native-space pred
        scores = out[0][:, 4].cpu().numpy() 
        pitchs_yaws_rolls = out[0][:, 6:].cpu().numpy()   # N*3
        for i, [x1, y1, x2, y2] in enumerate(bboxes):
            im0 = cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), 
                [255,255,255], thickness=args.thickness)
            # im0 = cv2.putText(im0, str(round(scores[i], 3)), (int(x1), int(y1)), 
                # cv2.FONT_HERSHEY_PLAIN, 0.7, (255,255,255), thickness=2)
            pitch = (pitchs_yaws_rolls[i][0] - 0.5) * 180
            yaw = (pitchs_yaws_rolls[i][1] - 0.5) * 360
            roll = (pitchs_yaws_rolls[i][2] - 0.5) * 180
            im0 = plot_3axis_Zaxis(im0, yaw, pitch, roll, tdx=(x1+x2)/2, tdy=(y1+y2)/2, 
                size=max(y2-y1, x2-x1)*0.8, thickness=args.thickness)
                
        cv2.imwrite(single_path[:-4]+"_res.jpg", im0)
        
  