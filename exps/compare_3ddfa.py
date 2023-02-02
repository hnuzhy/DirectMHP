
__author__ = 'Huayi Zhou'

'''
Put this file under the main folder of codes project 3DDFA https://github.com/cleardusk/3DDFA

usage:
python compare_3ddfa.py --root-imgdir /path/to/root/imgdir \
    --json-file /path/to/prepared/json/file \
    --save-file /path/to/saving/npy/file -m gpu

e.g.:
python compare_3ddfa.py --root-imgdir /datasdc/zhouhuayi/dataset/AGORA/HPE/images/validation --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/agora_m_1280_e300_t40_lw010/weights/val_best_c0.001_i0.65_pd_frontal.json --save-file ./agora_val_3DDFA.npy -m gpu --debug false
[results]
Saving all results in one file ./agora_val_3DDFA.npy ...
Inference one image taking time: 0.011305771888510957
face number: 3403 / 3403; MAE: 48.5867, [pitch_error, yaw_error, roll_error]: 42.5566, 39.6174, 63.5861


python compare_3ddfa.py --root-imgdir /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/images/val --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/cmu_m_1280_e200_t40_lw010/weights/val_best_c0.001_i0.65_pd_frontal.json --save-file ./cmu_val_3DDFA.npy -m gpu --debug false
[results]
Saving all results in one file ./cmu_val_3DDFA.npy ...
Inference one image taking time: 0.017735703712104287
face number: 15871 / 15871; MAE: 27.1172, [pitch_error, yaw_error, roll_error]: 26.3376, 23.3927, 31.6214

'''

import os
import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import argparse
import torch.backends.cudnn as cudnn
import time
from tqdm import tqdm
import json

from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
from utils.inference import parse_roi_box_from_landmark, \
    crop_img, predict_68pts, parse_roi_box_from_bbox, predict_dense
from utils.estimate_pose import parse_pose, parse_pose_v2
from utils.cv_plot import plot_pose_box

STD_SIZE = 120

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def main(args):

    # 1. load pre-tained model
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()


    # 2. forward
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    
    with open(args.json_file, "r") as json_f:
        pd_results_list = json.load(json_f)
     
    # face_imgs = []  # cropped face images collection
    pts_res = []  # 3d facial landmarks collection
    camPs = []  # Camera matrix collection
    pd_poses = []  # predicted pose collection
    gt_poses = []  # ground-truth pose collection
    taking_time_list = []  # how many ms per face
    valid_face_num = 0
    for ind, pd_results in enumerate(tqdm(pd_results_list)):
        if args.debug and ind > 50: break  # for testing
        
        img_path = os.path.join(args.root_imgdir, str(pd_results["image_id"])+".jpg")
        img_ori = cv2.imread(img_path)
        
        bbox = pd_results['bbox'] # bbox default format is [x0,y0,w,h], should be converted to [x0,y0,x1,y1]
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

        gt_pitch = pd_results['gt_pitch']
        gt_yaw = pd_results['gt_yaw']
        gt_roll = pd_results['gt_roll']
        
        ''' We do not need this enlarge operation. Or results will be super bad.'''
        # roi_box = parse_roi_box_from_bbox(bbox)  
        roi_box = bbox
        img = crop_img(img_ori, roi_box)
        
        t1 = time.time()
        # forward: one step
        img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input = transform(img).unsqueeze(0)
        with torch.no_grad():
            if args.mode == 'gpu':
                input = input.cuda()
            param = model(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
        
        # 68 pts
        pts68 = predict_68pts(param, roi_box)

        ''' two-step for more accurate bbox to crop face '''
        # roi_box = parse_roi_box_from_landmark(pts68)
        # img_step2 = crop_img(img_ori, roi_box)
        # img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        # input = transform(img_step2).unsqueeze(0)
        # with torch.no_grad():
            # if args.mode == 'gpu':
                # input = input.cuda()
            # param = model(input)
            # param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
        # pts68 = predict_68pts(param, roi_box)
            
        t2 = time.time()
        taking_time_list.append(t2-t1)
        
        camP, pose = parse_pose(param)
        # camP, pose = parse_pose_v2(param, pts68)
        if pose is None:
            continue
        
        valid_face_num += 1
        pts_res.append(pts68)
        camPs.append(camP)
        
        # the predicted order of 3DDFA is: [yaw, -pitch, -roll], and in range (-np.pi/2, np.pi/2) 
        pd_poses.append([-pose[1]*180/np.pi, pose[0]*180/np.pi, -pose[2]*180/np.pi])  # for parse_pose()
        # pd_poses.append([pose[1], pose[0], pose[2]])  # for parse_pose_v2()
        gt_poses.append([gt_pitch, gt_yaw, gt_roll])
        

        if args.debug:
            save_img_path = "./tmp/"+str(ind).zfill(0)+\
                "_p"+str(round(gt_pitch, 2))+"v"+str(round(-pose[1]*180/np.pi, 2))+\
                "_y"+str(round(gt_yaw, 2))+"v"+str(round(pose[0]*180/np.pi, 2))+\
                "_r"+str(round(gt_roll, 2))+"v"+str(round(-pose[2]*180/np.pi, 2))+".jpg"  # for parse_pose()
            # save_img_path = "./tmp/"+str(ind).zfill(0)+\
                # "_p"+str(round(gt_pitch, 2))+"v"+str(round(pose[1], 2))+\
                # "_y"+str(round(gt_yaw, 2))+"v"+str(round(pose[0], 2))+\
                # "_r"+str(round(gt_roll, 2))+"v"+str(round(pose[2], 2))+".jpg"  # for parse_pose_v2()
            

            cv2.rectangle(img_ori, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)
            for i in range(len(pts68[0, :])):
                cv2.circle(img_ori, (int(pts68[0, i]), int(pts68[1, i])), 1, (0,255,255), -1)
            img_ori = plot_pose_box(img_ori, [camP], [pts68])
            cv2.imwrite(save_img_path, img_ori)
            
            

    '''print all results'''
    print("Saving all results in one file %s ..."%(args.save_file))
    np.savez(args.save_file, camPs=np.array(camPs), 
        pts_res=np.array(pts_res),
        # image=np.array(face_imgs), 
        pd_pose=np.array(pd_poses), 
        gt_poses=np.array(gt_poses))
    # db_dict = np.load(args.save_file)
    # print(args.save_file, list(db_dict.keys()))
   
   
    print("Inference one image taking time:", sum(taking_time_list)/len(taking_time_list))
    
    
    error_list = np.abs(np.array(gt_poses) - np.array(pd_poses))
    # error_list[:, 1] = np.min((error_list[:, 1], 360 - error_list[:, 1]), axis=0)  # yaw range may be [-180,180]
    error_list = np.min((error_list, 360 - error_list), axis=0)
    pose_matrix = np.mean(error_list, axis=0)
    MAE = np.mean(pose_matrix)
    print("face number: %d / %d; MAE: %s, [pitch_error, yaw_error, roll_error]: %s, %s, %s"%(
        valid_face_num, len(taking_time_list), round(MAE, 4), 
        round(pose_matrix[0], 4), round(pose_matrix[1], 4), round(pose_matrix[2], 4)))
        
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    
    parser.add_argument('--root-imgdir', default='',
                        help='root path to multiple images')
    parser.add_argument('--json-file', default='',
                        help='json file path that contains multiple images and their head bboxes')
    parser.add_argument('--save-file', default='',
                        help='.npy file path to save all results')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--debug', default='false', type=str2bool, help='whether set into debug mode')
    
    args = parser.parse_args()
    main(args)