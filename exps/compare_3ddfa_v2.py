
__author__ = 'Huayi Zhou'

'''

pip install onnxruntime

Put this file under the main folder of codes project 3DDFA_v2 https://github.com/cleardusk/3DDFA_V2
or
Put this file under the main folder of codes using project 3DDFA_v2 https://github.com/bubingy/HeadPoseEstimate

usage:
python compare_3ddfa_v2.py --root-imgdir /path/to/root/imgdir \
    --json-file /path/to/prepared/json/file \
    --save-file /path/to/saving/npy/file -m gpu

e.g.:
python compare_3ddfa_v2.py --root-imgdir /datasdc/zhouhuayi/dataset/AGORA/HPE/images/validation --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/agora_m_1280_e300_t40_lw010/weights/val_best_c0.001_i0.65_pd_frontal.json --save-file ./agora_val_3DDFA_v2.npy -m gpu --debug
[results]
Saving all results in one file ./agora_val_3DDFA_v2.npy ...
Inference one image taking time: 0.015800806553336474
face number: 3403; MAE: 22.7539, [pitch_error, yaw_error, roll_error]: 20.5154, 28.4544, 19.2918


python compare_3ddfa_v2.py --root-imgdir /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/images/val --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/cmu_m_1280_e200_t40_lw010/weights/val_best_c0.001_i0.65_pd_frontal.json --save-file ./cmu_val_3DDFA_v2.npy -m gpu --debug
[results]
Saving all results in one file ./cmu_val_3DDFA_v2.npy ...
Inference one image taking time: 0.016364026179303746
face number: 15871; MAE: 17.3448, [pitch_error, yaw_error, roll_error]: 18.6524, 17.0074, 16.3747

'''

import cv2
import os
import time
import json
import argparse
from tqdm import tqdm
import numpy as np

from model.pose import estimate_head_pose
from model.plot import draw_pose

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def main(args):

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '1'
    from model.FaceAlignment3D.TDDFA_ONNX import TDDFA_ONNX
    tddfa = TDDFA_ONNX()

    with open(args.json_file, "r") as json_f:
        pd_results_list = json.load(json_f)
     

    pts_res = []  # 3d facial landmarks collection
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
        
        tic = time.time()
        
        param_lst, roi_box_lst = tddfa(img_ori, [bbox])
        
        # calculate Euler angle
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst)
        euler_angle_lst, directions_lst, landmarks_lst = estimate_head_pose(ver_lst, True)
        
        toc = time.time()
        taking_time_list.append(toc-tic)

        pts_res.append(landmarks_lst[0])
        pose = euler_angle_lst[0]
        
        # the predicted order of 3DDFA_v2 is: [-roll, -yaw, -pitch]
        pose[:] = -pose[:]
        
        pd_poses.append([pose[2], pose[1], pose[0]])
        gt_poses.append([gt_pitch, gt_yaw, gt_roll])
        

        if args.debug:
            save_img_path = "./tmp/"+str(ind).zfill(0)+\
                "_p"+str(round(gt_pitch, 2))+"v"+str(round(pose[2], 2))+\
                "_y"+str(round(gt_yaw, 2))+"v"+str(round(pose[1], 2))+\
                "_r"+str(round(gt_roll, 2))+"v"+str(round(pose[0], 2))+".jpg"
                
            # cv2.rectangle(img_ori, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)

            show_img = draw_pose(img_ori, directions_lst, np.array([bbox]), landmarks_lst,
                show_bbox=True, show_landmarks=True)
            cv2.imwrite(save_img_path, show_img)

    '''print all results'''
    print("Saving all results in one file %s ..."%(args.save_file))
    np.savez(args.save_file,
        pts_res=np.array(pts_res),
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
    print("face number: %d; MAE: %s, [pitch_error, yaw_error, roll_error]: %s, %s, %s"%(len(taking_time_list), 
        round(MAE, 4), round(pose_matrix[0], 4), round(pose_matrix[1], 4), round(pose_matrix[2], 4)))
        
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    
    parser.add_argument('--root-imgdir', default='',
                        help='root path to multiple images')
    parser.add_argument('--json-file', default='',
                        help='json file path that contains multiple images and their head bboxes')
    parser.add_argument('--save-file', default='',
                        help='.npy file path to save all results')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--debug',  action='store_true', help='whether set into debug mode')
    
    args = parser.parse_args()
    main(args)