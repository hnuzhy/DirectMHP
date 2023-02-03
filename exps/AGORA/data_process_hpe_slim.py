
'''Details are al in readme.txt file'''
import os
import copy
import pandas
import cv2
import json
import shutil

# import pickle
import pickle5 as pickle
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from hpe_utils import projectPoints
from hpe_utils import align_3d_head
from hpe_utils import reference_head
from hpe_utils import get_sphere
from hpe_utils import select_euler
from hpe_utils import inverse_rotate_zyx

from hpe_utils import draw_axis, plot_pose_cube, plot_3axis_Zaxis_by_euler_angles

############################################################################################

# AGORA

skeleton_joints = [1,2,4,5,7,8,12,15,16,17,18,19,20,21]  # selected 14 joints index from 25 given joints

limbs_connect = [[1,4], [4,7], [2,5], [5,8], [1,12], [2,12], [12,15], 
    [12,16], [12,17], [16,18], [18,20], [17,19], [19,21]]

coco_style_hpe_dict = {
    'info': {
        'description': 'Face landmarks and Euler angles of AGORA Dataset',
        'url': 'https://agora.is.tue.mpg.de/',
        'version': '1.0',
        'year': 2022,
        'contributor': 'Huayi Zhou',
        'date_created': '2022/02/18',
    },
    'licences': [{
        'url': 'http://creativecommons.org/licenses/by-nc/2.0',
        'name': 'Attribution-NonCommercial License'
    }],
    'images': [],
    'annotations': [],
    'categories': [{
        'supercategory': 'person',
        'id': 1,
        'name': 'person'
    }]
}

face_edges = np.array([  # 51 landmarks
    [ 0,  1],[ 1,  2],[ 2,  3],[ 3,  4], #right eyebrow
    [ 5,  6],[ 6,  7],[ 7,  8],[ 8,  9], #left eyebrow
    [10, 11],[11, 12],[12, 13],   #nose upper part
    [14, 15],[15, 16],[16, 17],[17, 18], #nose lower part
    [19, 20],[20, 21],[21, 22],[22, 23],[23, 24],[24, 19], #right eye
    [25, 26],[26, 27],[27, 28],[28, 29],[29, 30],[30, 25], #left eye
    [31, 32],[32, 33],[33, 34],[34, 35],[35, 36],[36, 37],[37, 38],[38, 39],[39, 40],[40, 41],[41, 42],[42, 31], #Lip outline
    [43, 44],[44, 45],[45, 46],[46, 47],[47, 48],[48, 49],[49, 50],[50, 43] #Lip inner line 
    ])
    
coco_style_dict_val = copy.deepcopy(coco_style_hpe_dict)
coco_style_dict_train = copy.deepcopy(coco_style_hpe_dict)

############################################################################################

# init transform params

E_ref = np.mat([[1, 0, 0, 0.],
            [0, -1, 0, 0],
            [0, 0, -1, 50],
            [0, 0, 0,  1]])

model_points, _ = reference_head(scale=1, pyr=(0., 0., 0.))
model_points_3D = np.ones((4, 58), dtype=np.float32)
model_points_3D[0:3] = model_points

# kp_idx = np.asarray([17, 21, 26, 22, 45, 42, 39, 36, 35, 31, 54, 48, 57, 8])  # 14 indexs of refered points in CMUPanoptic
# kp_idx_model = np.asarray([38, 34, 33, 29, 13, 17, 25, 21, 54, 50, 43, 39, 45, 6])  # 14 indexs of refered points in FaceModel

kp_idx_agora = np.asarray([0, 4, 9, 5, 28, 25, 22, 19, 18, 14, 37, 31, 40])  # 13 indexs of refered points in AGORA
kp_idx_model = np.asarray([38, 34, 33, 29, 13, 17, 25, 21, 54, 50, 43, 39, 45])  # 13 indexs of refered points in FaceModel

kp_idx_agora_10 = np.asarray([ 0,  9, 25, 22, 18, 14, 37, 31, 40])  # 9
kp_idx_model_10 = np.asarray([38, 33, 17, 25, 54, 50, 43, 39, 45])  # 9

kp_idx_agora_18 = np.asarray([ 0,  2,  4,  9,  7,  5, 28, 25, 22, 19, 18, 16, 14, 37, 31, 34, 40])  # 17
kp_idx_model_18 = np.asarray([38, 36, 34, 33, 31, 29, 13, 17, 25, 21, 54, 52, 50, 43, 39, 41, 45])  # 17


sphere = []
for theta in range(0, 360, 10):
    for phi in range(0, 180, 10):
        # sphere.append(get_sphere(theta, phi, 22))  # default radius is 22
        sphere.append(get_sphere(theta, phi, 18))
sphere = np.asarray(sphere)
sphere = sphere + [0, 5, -5]
sphere = sphere.T

img_w, img_h = 1280, 720

############################################################################################

def plot_agora_joints_vis(img, occlusion, face, scale, remove=False):
    
    for i in range(len(face)):
        px, py = int(face[i, 0]*scale), int(face[i, 1]*scale)

        if remove:
            img = cv2.circle(img, (px, py), 1, (0,0,255), -1)
        else:
            img = cv2.circle(img, (px, py), 1, (0,255,255), -1)
            img = cv2.putText(img, str(i), (px, py), cv2.FONT_HERSHEY_PLAIN, 0.6, (255,255,255), thickness=1)
 
    return img

    
def auto_labels_generating(imgPath, filter_joints_list):
    
    valid_bbox_euler_list = []

    lost_faces = 0
    for [face2d, occlusion, face3d, camR, camT, camK] in filter_joints_list: 
        # face3d has 51 3D joints, stored as an array with shape [51,3]
        face3d = np.array(face3d).reshape((-1, 3)).transpose()

        rotation, translation, error, scale = align_3d_head(
            np.mat(model_points_3D[0:3, kp_idx_model]), np.mat(face3d[:, kp_idx_agora]))
        
        sphere_new = scale * rotation @ (sphere) + translation
        pt_helmet = projectPoints(sphere_new, camK, camR, camT, [0,0,0,0,0])
            
        temp = np.zeros((4, 4))
        temp[0:3, 0:3] = rotation
        temp[0:3, 3:4] = translation
        temp[3, 3] = 1
        E_virt = np.linalg.inv(temp @ np.linalg.inv(E_ref))
        
        E_real = np.zeros((4, 4))
        E_real[0:3, 0:3] = camR
        E_real[0:3, 3:4] = camT
        E_real[3, 3] = 1

        compound = E_real @ np.linalg.inv(E_virt)
        status, [pitch, yaw, roll] = select_euler(np.rad2deg(inverse_rotate_zyx(compound)))
        yaw = -yaw
        roll = -roll
        
        
        rotation_10, translation_10, _, _ = align_3d_head(
            np.mat(model_points_3D[0:3, kp_idx_model_10]), np.mat(face3d[:, kp_idx_agora_10]))
        temp[0:3, 0:3] = rotation_10
        temp[0:3, 3:4] = translation_10
        E_virt = np.linalg.inv(temp @ np.linalg.inv(E_ref))
        compound = E_real @ np.linalg.inv(E_virt)
        status_10, [pitch_10, yaw_10, roll_10] = select_euler(np.rad2deg(inverse_rotate_zyx(compound)))
        yaw_10 = -yaw_10
        roll_10 = -roll_10
        
        rotation_18, translation_18, _, _ = align_3d_head(
            np.mat(model_points_3D[0:3, kp_idx_model_18]), np.mat(face3d[:, kp_idx_agora_18]))
        temp[0:3, 0:3] = rotation_18
        temp[0:3, 3:4] = translation_18
        E_virt = np.linalg.inv(temp @ np.linalg.inv(E_ref))
        compound = E_real @ np.linalg.inv(E_virt)
        status_18, [pitch_18, yaw_18, roll_18] = select_euler(np.rad2deg(inverse_rotate_zyx(compound)))
        yaw_18 = -yaw_18
        roll_18 = -roll_18
        
        
        if status == True:
            x_min = int(max(min(pt_helmet[0, :]),0))
            y_min = int(max(min(pt_helmet[1, :]),0))
            x_max = int(min(max(pt_helmet[0, :]), img_w))
            y_max = int(min(max(pt_helmet[1, :]), img_h))
            w, h = x_max-x_min, y_max-y_min
            '''
            Exclude heads with "out-of-bounding position", "severely truncated" or "super-large size".
            However, we still could not filter out heads "without face labels" or "totally occluded".
            '''
            if x_min<x_max and y_min<y_max and h/w<1.5 and w/h<1.5 and w<img_w*0.7 and h<img_h*0.7:  # sanity check
                head_bbox = [x_min, y_min, w, h]  # format [x,y,w,h]
                euler_angles = [pitch, yaw, roll]  # represented by degree
                valid_bbox_euler_list.append({
                    "face2d_pts": [list(face2d[:,0]), list(face2d[:,1])], 
                    "head_bbox": head_bbox, 
                    "euler_angles_10": [pitch_10, yaw_10, roll_10] if status_10 == True else [],
                    "euler_angles_18": [pitch_18, yaw_18, roll_18] if status_18 == True else [],
                    "euler_angles": euler_angles})
            else:
                # print("The face in this frame is not having valid face bounding box...")
                continue
        else:
            # print("The face in this frame is not having valid three Euler angles...")
            lost_faces += 1
            continue

    return valid_bbox_euler_list, lost_faces
    

def parse_pkl_file(data, type, index, 
    images_folder, images_vis_folder, images_save_folder, mapping_dict, debug):

    for idx in tqdm(range(len(data))):
        cur_image_path = data.iloc[idx].at['imgPath']
        cur_valid_flag = data.iloc[idx].at['isValid']  # bool list, True or False
        cur_occlusion = data.iloc[idx].at['occlusion']  # float list, value in range [0,100] 
        cur_kid = data.iloc[idx].at['kid']  # bool list, True or False
        cur_age = data.iloc[idx].at['age']  # string list, e.g., '31-50', '50+'
        cur_joints_list = data.iloc[idx].at['gt_joints_2d']  # numpy array list, length is 127
        # cur_joints_3d_list = data.iloc[idx].at['gt_joints_3d']  # numpy array list, length is 127
        cur_joints_3d_list = data.iloc[idx].at['cam_j3d']  # numpy array list, length is 127
        cur_camR_list = data.iloc[idx].at['camR']  # numpy array list, camera extrinsics R (3x3)
        cur_camT_list = data.iloc[idx].at['camT']  # numpy array list, camera extrinsics T (3x1)
        cur_camK_list = data.iloc[idx].at['camK']  # numpy array list, camera intrinsics K (3x3)
        
        if type == "train":
            image_path = os.path.join(images_folder+"_"+str(index), cur_image_path)
        else:
            image_path = os.path.join(images_folder, cur_image_path)
        
        if debug and idx == 0:
            print(index, "\t", idx, "\t", cur_image_path)
            print(cur_valid_flag, "\n", cur_occlusion, "\n", cur_kid, "\n", cur_age)
            print(len(cur_joints_list), "\t", len(cur_joints_list[0]))
        
        '''
        'gt_joints_2d' foramt are [15-skeleton_keypoints + 2*21-hand_keypoints + 70-face_keypoint], 127 joints
        Details are in OpenPose Repo https://github.com/CMU-Perceptual-Computing-Lab/openpose#features
        https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html
        
        Real Order of 'gt_joints_2d' foramt (from official AGORA website)
        https://github.com/pixelite1201/agora_evaluation/blob/master/agora_evaluation/calculate_v2v_error.py#L301
        skeleton, hands, face = joints[:25], joints[25:56], joints[56:]
        face = joints[56:]  # not real condition when vis using debug mode (total --> 127-56=71 joints)
        face = joints[56+4+10+6:]  # 4 another unknown pts of head; 10 fingers joints; 6 foot joints (total --> 51 joints)
        '''
        filter_joints_list, remove_joints_list = [], []
        for ind, (isValid, occlusion, kid, age, joints, joints_3d, camR, camT, camK) in enumerate(
            zip(cur_valid_flag, cur_occlusion, cur_kid, cur_age, cur_joints_list, 
                cur_joints_3d_list, cur_camR_list, cur_camT_list, cur_camK_list)):
            
            face, face_3d = joints[56+4+10+6:], joints_3d[56+4+10+6:]
            
            if debug:
                print(index, cur_image_path, ind, isValid, occlusion, kid, age, len(joints), len(face))
            
            if occlusion < 90:
                # filter_joints_list.append([skeleton, hands, face, occlusion])
                filter_joints_list.append([face, occlusion, face_3d, camR, camT, camK])
            else:
                # remove_joints_list.append([skeleton, hands, face, occlusion])
                remove_joints_list.append([face, occlusion, face_3d, camR, camT, camK])
         
        valid_bbox_euler_list, lost_faces = auto_labels_generating(cur_image_path, filter_joints_list)
            
        if debug:
            img = cv2.imread(image_path)
            
            # cv2.putText(img, "lost_faces:"+str(lost_faces), (10,40), 
                    # cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), thickness=2)
            for bbox_euler in valid_bbox_euler_list:
                head_bbox, euler_angles = bbox_euler["head_bbox"], bbox_euler["euler_angles"]
                [x_min, y_min, w, h] = head_bbox
                [pitch, yaw, roll] = euler_angles
                img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_min+w), int(y_min+h)), (255,255,255), 2)
                img = draw_axis(img, yaw, pitch, roll, tdx=x_min+w/2, tdy=y_min+h/2, size=60.0)
                # img = plot_3axis_Zaxis_by_euler_angles(img, yaw, pitch, roll, 
                    # tdx=x_min+w/2, tdy=y_min+h/2, size=40.0,limited=True)
                
                # face2d = bbox_euler["face2d_pts"]
                # fpx, fpy = int(sum(face2d[0])/len(face2d[0])), int(sum(face2d[1])/len(face2d[1]))
                # img = cv2.line(img, (fpx, fpy), (int(x_min), int(y_min)), (0,0,0), 2)
                for fai, fbi in face_edges:
                    ptxa, ptya = bbox_euler["face2d_pts"][0][fai], bbox_euler["face2d_pts"][1][fai]
                    ptxb, ptyb = bbox_euler["face2d_pts"][0][fbi], bbox_euler["face2d_pts"][1][fbi]
                    cv2.line(img, (int(ptxa), int(ptya)), (int(ptxb), int(ptyb)), (255,255,0), 2)
                
            # scale = 4.0
            # img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            # for filter_joints in filter_joints_list:
                # [face, occlusion, face_3d, camR, camT, camK] = filter_joints
                # img = plot_agora_joints_vis(img, occlusion, face, scale, remove=False)
            # for remove_joints in remove_joints_list:
                # [face, occlusion, face_3d, camR, camT, camK] = remove_joints
                # img = plot_agora_joints_vis(img, occlusion, face, scale, remove=True)
                
            save_path = os.path.join(images_vis_folder, cur_image_path[:-4]+".jpg")  # jpg is smaller than png
            cv2.imwrite(save_path, img)
            
            if idx > 10: break

        else:
            '''begin to process original labels of AGORA'''
            seq_name = cur_image_path.replace("_1280x720.png", "")  # seq_key_5_15_xxxxx_1280x720.png
            cur_frame = int(seq_name[-5:])
            
            seq_key = seq_name[:-6]
            if seq_key in mapping_dict[type]: mapping_dict[type][seq_key] += 1
            else: mapping_dict[type][seq_key] = 1
            seq_key_list = list(mapping_dict[type].keys())
            seq_ind = seq_key_list.index(seq_key) + 1
            
            # 1yyyyxxxxx for train, 2yyyyxxxxx for validation
            if type == "train":
                image_id = 1000000000 + seq_ind*100000 + cur_frame
            if type == "validation":
                image_id = 2000000000 + seq_ind*100000 + cur_frame
            
            '''coco_style_sample'''
            image_dict = {
                'file_name': str(image_id)+".jpg",
                'height': img_h,
                'width': img_w,
                'id': image_id,
                'seq_key': seq_key,
            }
            temp_annotations_list = []
            for ind, labels in enumerate(valid_bbox_euler_list):
                temp_annotation = {
                    # 'face2d_pts': labels["face2d_pts"],
                    'bbox': labels["head_bbox"],  # please use the default 'bbox' as key in cocoapi
                    'euler_angles': labels["euler_angles"],  # with 13 reference points
                    'euler_angles_10': labels["euler_angles_10"],  # with 9 reference points
                    'euler_angles_18': labels["euler_angles_18"],  # with 17 reference points
                    'image_id': image_id,
                    'id': image_id * 100 + ind,  # we support that no image has more than 100 persons/poses
                    'category_id': 1,
                    'iscrowd': 0,
                    # 'segmentation': [],  # This script is not for segmentation
                    # 'area': round(labels["head_bbox"][-1] * labels["head_bbox"][-2], 4)
                }
                temp_annotations_list.append(temp_annotation)
                
            if len(temp_annotations_list) != 0:
                if type == "train":
                    coco_style_dict_train['images'].append(image_dict)
                    coco_style_dict_train['annotations'] += temp_annotations_list
                if type == "validation":
                    coco_style_dict_val['images'].append(image_dict)
                    coco_style_dict_val['annotations'] += temp_annotations_list

                dst_img_path = os.path.join(images_save_folder, str(image_id)+".jpg")
                shutil.copy(image_path, dst_img_path)
            else:
                continue  # after processing, this cur_frame with Null json annotation, skip it

        # finish one image/frame
    return mapping_dict


if __name__ == "__main__":
    
    debug = False  # True or False, set as True will plot labels on partial images for visualization
    has_parsed = False  # True or False, set as True will not operate parse_pkl_file() for saving time
    load_pkl_flag = True # True or False, set as True will always load pkl files (taking much time)
    
    mapping_dict = {"train": {}, "validation": {}}
    euler_angles_stat = [[],[],[]]  # pitch, yaw, roll
    
    gt_frontal_face_num = {"train": 0, "validation": 0}
    
    # for type in ["validation", "train"]:
    # for type in ["validation"]:
    for type in ["train"]:
        images_folder = "./demo/images/%s"%(type)
        images_vis_folder = "./demo/images/%s_vis_face"%(type)
        
        images_save_folder = "./HPE/images/%s"%(type)
        anno_save_folder = "./HPE/annotations/coco_style_%s_slim.json"%(type)
        
        print("\nGenerating %s set ..."%(type))
        
        total_images, labeled_images = 0, 0
        if type == "validation": total_images = len(os.listdir(images_folder))
        
        index_list = [0,1,2,3,4,5,6,7,8,9]
        for index in index_list:
            pkl_withjv_path = "./demo/Cam/%s_Cam/%s_%d_withjv.pkl"%(type, type, index)
            print(pkl_withjv_path)
            
            if not load_pkl_flag:
                continue
            
            data = pickle.load(open(pkl_withjv_path, 'rb'), encoding='iso-8859-1')
            if index ==0: print(list(data.keys()))
            labeled_images += len(data)
            
            if type == "train": total_images += len(os.listdir(images_folder+"_"+str(index)))
            
            if not has_parsed:
                mapping_dict = parse_pkl_file(data, type, index, 
                    images_folder, images_vis_folder, images_save_folder, mapping_dict, debug)
            
        if type == "train":
            if not has_parsed and not debug:
                print(len(mapping_dict["train"]), "\n", mapping_dict["train"])
                with open(anno_save_folder, "w") as dst_ann_file:
                    json.dump(coco_style_dict_train, dst_ann_file)
            else:
                with open(anno_save_folder, "r") as dst_ann_file:
                    coco_style_dict_train = json.load(dst_ann_file)
            print("\ntrain: original images-->%d, labeled images-->%d, left images-->%d, left instances-->%d"%(
                total_images, labeled_images, len(coco_style_dict_train['images']), len(coco_style_dict_train['annotations'])))
            
            for labels_dict in coco_style_dict_train['annotations']:
                [pitch, yaw, roll] = labels_dict['euler_angles']
                euler_angles_stat[0].append(pitch)
                euler_angles_stat[1].append(yaw)
                euler_angles_stat[2].append(roll)
                
                if abs(yaw) < 90: gt_frontal_face_num["train"] += 1
                
                if 'head_bbox' in labels_dict:
                    labels_dict['bbox'] = labels_dict['head_bbox']  # please use the default 'bbox' as key in cocoapi
                    del labels_dict['head_bbox']
                if 'area' not in labels_dict:  # generate standard COCO style json file
                    labels_dict['segmentation'] = []  # This script is not for segmentation
                    labels_dict['area'] = round(labels_dict['bbox'][-1] * labels_dict['bbox'][-2], 4)
            with open(anno_save_folder, "w") as dst_ann_file:  # rewrite coco_style_dict_train into its json file
                json.dump(coco_style_dict_train, dst_ann_file)
            
        if type == "validation":
            if not has_parsed and not debug:
                print(len(mapping_dict["validation"]), "\n", mapping_dict["validation"])
                with open(anno_save_folder, "w") as dst_ann_file:
                    json.dump(coco_style_dict_val, dst_ann_file)
            else:
                with open(anno_save_folder, "r") as dst_ann_file:
                    coco_style_dict_val = json.load(dst_ann_file)
            print("\nvalidation: original images-->%d, labeled images-->%d, left images-->%d, left instances-->%d"%(
                total_images, labeled_images, len(coco_style_dict_val['images']), len(coco_style_dict_val['annotations'])))
                
            for labels_dict in coco_style_dict_val['annotations']:
                [pitch, yaw, roll] = labels_dict['euler_angles']
                euler_angles_stat[0].append(pitch)
                euler_angles_stat[1].append(yaw)
                euler_angles_stat[2].append(roll)
                
                if abs(yaw) < 90: gt_frontal_face_num["validation"] += 1
                
                if 'head_bbox' in labels_dict:
                    labels_dict['bbox'] = labels_dict['head_bbox']  # please use the default 'bbox' as key in cocoapi
                    del labels_dict['head_bbox']
                if 'area' not in labels_dict:  # generate standard COCO style json file
                    labels_dict['segmentation'] = []  # This script is not for segmentation
                    labels_dict['area'] = round(labels_dict['bbox'][-1] * labels_dict['bbox'][-2], 4)
            with open(anno_save_folder, "w") as dst_ann_file:  # rewrite coco_style_dict_val into its json file
                json.dump(coco_style_dict_val, dst_ann_file)
            
    print("frontal_face_num:\t", gt_frontal_face_num)       
    

    ''' (7505+105046)/(1070+14408) = 112551/15478 = 7.27 instances/per_image
    
    5
     {'ag_validationset_renderpeople_bfh_flowers_5_15': 259, 'ag_validationset_renderpeople_bfh_brushifyforest_5_15': 180, 'ag_validationset_renderpeople_bfh_brushifygrasslands_5_15': 183, 'ag_validationset_renderpeople_bfh_hdri_50mm_5_15': 196, 'ag_validationset_renderpeople_bfh_archviz_5_10_cam02': 259}

    validation: original images-->1225, labeled images-->1077, left images-->1070, left instances-->7505
    frontal_face_num:        {'train': 0, 'validation': 3781}
    
    54
    {'ag_trainset_renderpeople_body_hdri_50mm_5_10': 401, 'ag_trainset_renderpeople_bfh_archviz_5_10_cam02': 1119, 'ag_trainset_renderpeople_bfh_brushifyforest_5_15': 943, 'ag_trainset_renderpeople_bfh_brushifygrasslands_5_15': 927, 'ag_trainset_axyz_bfh_construction_5_15': 341, 'ag_trainset_axyz_bfh_flowers_5_15': 372, 'ag_trainset_axyz_bfh_hdri_50mm_5_10': 424, 'ag_trainset_humanalloy_body_archviz_5_10_cam01': 45, 'ag_trainset_humanalloy_body_brushifyforest_5_15': 38, 'ag_trainset_humanalloy_body_brushifygrasslands_5_15': 41, 'ag_trainset_humanalloy_body_construction_5_15': 29, 'ag_trainset_humanalloy_body_flowers_5_15': 39, 'ag_trainset_humanalloy_body_hdri_50mm_5_10': 38, 'ag_trainset_humanalloy_bfh_archviz_5_10_cam01': 197, 'ag_trainset_humanalloy_bfh_brushifyforest_5_15': 162, 'ag_trainset_humanalloy_bfh_brushifygrasslands_5_15': 166, 'ag_trainset_renderpeople_bfh_flowers_5_15': 785, 'ag_trainset_renderpeople_bfh_hdri_50mm_5_10': 873, 'ag_trainset_3dpeople_bfh_construction_5_15': 245, 'ag_trainset_3dpeople_bfh_flowers_5_15': 248, 'ag_trainset_3dpeople_bfh_hdri_50mm_5_10': 268, 'ag_trainset_axyz_body_archviz_5_10_cam03': 122, 'ag_trainset_axyz_body_brushifyforest_5_15': 104, 'ag_trainset_axyz_body_brushifygrasslands_5_15': 110, 'ag_trainset_axyz_body_construction_5_15': 97, 'ag_trainset_axyz_body_flowers_5_15': 92, 'ag_trainset_axyz_body_hdri_50mm_5_10': 97, 'ag_trainset_axyz_bfh_archviz_5_10_cam03': 500, 'ag_trainset_axyz_bfh_brushifyforest_5_15': 435, 'ag_trainset_axyz_bfh_brushifygrasslands_5_15': 459, 'ag_trainset_renderpeople_body_brushifyforest_5_15': 429, 'ag_trainset_renderpeople_body_brushifygrasslands_5_15': 451, 'ag_trainset_renderpeople_body_construction_5_15': 363, 'ag_trainset_renderpeople_body_flowers_5_15': 378, 'ag_trainset_humanalloy_bfh_construction_5_15': 115, 'ag_trainset_humanalloy_bfh_flowers_5_15': 134, 'ag_trainset_humanalloy_bfh_hdri_50mm_5_10': 148, 'ag_trainset_moverenderpeople_bfh_archviz_5_10_cam02': 40, 'ag_trainset_moverenderpeople_bfh_brushifyforest_5_15': 40, 'ag_trainset_moverenderpeople_bfh_brushifygrasslands_5_15': 40, 'ag_trainset_moverenderpeople_bfh_construction_5_15': 40, 'ag_trainset_moverenderpeople_bfh_flowers_5_15': 40, 'ag_trainset_moverenderpeople_bfh_hdri_50mm_5_10': 40, 'ag_trainset_renderpeople_body_archviz_5_10_cam02': 509, 'ag_trainset_3dpeople_body_archviz_5_10_cam00': 73, 'ag_trainset_3dpeople_body_brushifyforest_5_15': 67, 'ag_trainset_3dpeople_body_brushifygrasslands_5_15': 68, 'ag_trainset_3dpeople_body_construction_5_15': 51, 'ag_trainset_3dpeople_body_flowers_5_15': 57, 'ag_trainset_3dpeople_body_hdri_50mm_5_10': 63, 'ag_trainset_3dpeople_bfh_archviz_5_10_cam00': 325, 'ag_trainset_3dpeople_bfh_brushifyforest_5_15': 280, 'ag_trainset_3dpeople_bfh_brushifygrasslands_5_15': 290, 'ag_trainset_renderpeople_bfh_construction_5_15': 771}

    train: original images-->14529, labeled images-->14529, left images-->14408, left instances-->105046
    frontal_face_num:        {'train': 52639, 'validation': 0}
    '''