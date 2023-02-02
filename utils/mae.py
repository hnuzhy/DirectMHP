'''
__author__ = 'Huayi Zhou'
'''

import os
import json
import numpy as np
import copy
from tqdm import tqdm

def sort_labels_by_image_id(labels_list):
    images_labels_dict = {}
    for i, labels_dict in enumerate(labels_list):
        image_id = str(labels_dict['image_id'])
        if image_id in images_labels_dict.keys():
            images_labels_dict[image_id].append(labels_dict)
        else:
            images_labels_dict[image_id] = [labels_dict]
    return images_labels_dict


def sort_images_by_image_id(images_list):
    images_images_dict = {}
    for i, images_dict in enumerate(images_list):
        image_id = str(images_dict['id'])
        images_images_dict[image_id] = images_dict
    return images_images_dict


def calculate_bbox_iou(bboxA, bboxB, format='xyxy'):
    if format == 'xywh':  # xy is in top-left, wh is size
        [Ax, Ay, Aw, Ah] = bboxA[0:4]
        [Ax0, Ay0, Ax1, Ay1] = [Ax, Ay, Ax+Aw, Ay+Ah]
        [Bx, By, Bw, Bh] = bboxB[0:4]
        [Bx0, By0, Bx1, By1] = [Bx, By, Bx+Bw, By+Bh]
    if format == 'xyxy':
        [Ax0, Ay0, Ax1, Ay1] = bboxA[0:4]
        [Bx0, By0, Bx1, By1] = bboxB[0:4]
        
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        return crossArea/(areaA + areaB - crossArea)

def mean_absolute_error_calculate(gt_json_path, pd_json_path, frontal_face):
    matched_iou_threshold = 0.5  # our nms_iou_thre is 0.65
    score_threshold = 0.7
    gt_data, pd_data = [], []  # shapes of both should be N*3
    gt_data_frontal, pd_data_frontal = [], []  # shapes of both should be N*3
    gt_data_backward, pd_data_backward = [], []  # shapes of both should be N*3

    gt_json = json.load(open(gt_json_path, "r"))
    pd_json = json.load(open(pd_json_path, "r"))
    
    gt_labels_list = gt_json['annotations']
    gt_images_labels_dict = sort_labels_by_image_id(gt_labels_list)
    
    if frontal_face:
        gt_json_frontal_face = copy.deepcopy(gt_json)
        gt_json_frontal_face['images'] = []
        gt_json_frontal_face['annotations'] = []
        pd_json_frontal_face = []
        pd_json_full_face = []
        gt_images_images_dict = sort_images_by_image_id(gt_json['images'])
        appeared_image_id_list = []
        
    
    for pd_image_dict in tqdm(pd_json):  # these predicted bboxes have been sorted by scores
        score = pd_image_dict['score']
        if score < score_threshold:
            continue
            
        pd_bbox = pd_image_dict['bbox']
        pd_pitch = pd_image_dict['pitch']
        pd_yaw = pd_image_dict['yaw']
        pd_roll = pd_image_dict['roll']
        image_id = pd_image_dict['image_id']
        
        labels = gt_images_labels_dict[str(image_id)]
        max_iou, matched_index = 0, -1
        for i, label in enumerate(labels):
            gt_bbox = label['bbox']
            temp_iou = calculate_bbox_iou(pd_bbox, gt_bbox, format='xywh')
            if temp_iou > max_iou:
                max_iou = temp_iou
                matched_index = i
        if max_iou > matched_iou_threshold:
            [gt_pitch, gt_yaw, gt_roll] = labels[matched_index]['euler_angles']
            gt_data.append([gt_pitch, gt_yaw, gt_roll])
            pd_data.append([pd_pitch, pd_yaw, pd_roll])
            
            if abs(gt_yaw) > 90:
                gt_data_backward.append([gt_pitch, gt_yaw, gt_roll])
                pd_data_backward.append([pd_pitch, pd_yaw, pd_roll])
            else:
                gt_data_frontal.append([gt_pitch, gt_yaw, gt_roll])
                pd_data_frontal.append([pd_pitch, pd_yaw, pd_roll])
    
            if frontal_face:
                pd_image_dict['gt_pitch'] = gt_pitch
                pd_image_dict['gt_yaw'] = gt_yaw
                pd_image_dict['gt_roll'] = gt_roll
                if abs(gt_yaw) < 90:
                    if str(image_id) not in appeared_image_id_list:
                        appeared_image_id_list.append(str(image_id))
                        gt_json_frontal_face['images'].append(gt_images_images_dict[str(image_id)])
                    gt_json_frontal_face['annotations'].append(labels[matched_index])
                    '''
                    This json file will be used for comparing with FAN, 3DDFA, FSA-Net and WHE-Net in narrow-range.
                    It will also be used for comparing with gt_json_frontal_face to calculate frontal bbox mAP.
                    '''
                    pd_json_frontal_face.append(pd_image_dict)  
                '''
                This json file will only be used for comparing with WHE-Net in full-range.
                '''
                pd_json_full_face.append(pd_image_dict)
                
                    
    total_num = len(gt_labels_list)
    left_num = len(gt_data)
    
    if left_num == 0:
        return total_num, [30,60,30], 2, [30,60,30], 1, [30,60,30], 1
    
    if frontal_face:
        with open(pd_json_path[:-5]+"_pd_frontal.json", 'w') as f:
            json.dump(pd_json_frontal_face, f)
        with open(pd_json_path[:-5]+"_gt_frontal.json", 'w') as f:
            json.dump(gt_json_frontal_face, f)
        with open(pd_json_path[:-5]+"_pd_full.json", 'w') as f:
            json.dump(pd_json_full_face, f)
            
            
    error_list = np.abs(np.array(gt_data) - np.array(pd_data))
    error_list[:, 1] = np.min((error_list[:, 1], 360 - error_list[:, 1]), axis=0)  # yaw range is [-180,180]
    pose_matrix = np.mean(error_list, axis=0)
    
    left_num_b = len(gt_data_backward)
    if left_num_b != 0:
        error_list_backward = np.abs(np.array(gt_data_backward) - np.array(pd_data_backward))
        error_list_backward[:, 1] = np.min((error_list_backward[:, 1], 360 - error_list_backward[:, 1]), axis=0)  # yaw range is [-180,180]
        pose_matrix_b = np.mean(error_list_backward, axis=0)
    else:
        pose_matrix_b, left_num_b = [30,60,30], 1

    left_num_f = len(gt_data_frontal)
    if left_num_f != 0:
        error_list_frontal = np.abs(np.array(gt_data_frontal) - np.array(pd_data_frontal))
        error_list_frontal[:, 1] = np.min((error_list_frontal[:, 1], 360 - error_list_frontal[:, 1]), axis=0)  # yaw range is [-180,180]
        pose_matrix_f = np.mean(error_list_frontal, axis=0)
    else:
        pose_matrix_f, left_num_f = [30,60,30], 1
        
    return total_num, pose_matrix, left_num, pose_matrix_b, left_num_b, pose_matrix_f, left_num_f
  

def mean_absolute_error_calculate_v2(gt_json_path, pd_json_path, frontal_face):
    matched_iou_threshold = 0.5  # our nms_iou_thre is 0.65
    score_threshold = 0.7
    gt_data, pd_data = [], []  # shapes of both should be N*3
    gt_data_frontal, pd_data_frontal = [], []  # shapes of both should be N*3
    gt_data_backward, pd_data_backward = [], []  # shapes of both should be N*3

    gt_json = json.load(open(gt_json_path, "r"))
    pd_json = json.load(open(pd_json_path, "r"))
    
    gt_labels_list = gt_json['annotations']
    # gt_images_labels_dict = sort_labels_by_image_id(gt_labels_list)
    pd_images_labels_dict = sort_labels_by_image_id(pd_json)
    
    if frontal_face:
        gt_json_frontal_face = copy.deepcopy(gt_json)
        gt_json_frontal_face['images'] = []
        gt_json_frontal_face['annotations'] = []
        pd_json_frontal_face = []
        pd_json_full_face = []
        gt_images_images_dict = sort_images_by_image_id(gt_json['images'])
        appeared_image_id_list = []
        
    
    for gt_label_dict in tqdm(gt_labels_list):  # matching for each GT label
        image_id = str(gt_label_dict['image_id'])
        gt_bbox = gt_label_dict['bbox']
        [gt_pitch, gt_yaw, gt_roll] = gt_label_dict['euler_angles']
        
        if image_id not in pd_images_labels_dict:  # this image has no bboxes been detected
            continue
            
        pd_results = pd_images_labels_dict[image_id]
        max_iou, matched_index = 0, -1
        for i, pd_result in enumerate(pd_results):  # match predicted bboxes in target image
            score = pd_result['score']
            if score < score_threshold:  # remove head bbox with low confidence
                continue
                
            pd_bbox = pd_result['bbox']
            temp_iou = calculate_bbox_iou(pd_bbox, gt_bbox, format='xywh')
            if temp_iou > max_iou:
                max_iou = temp_iou
                matched_index = i
                
        if max_iou > matched_iou_threshold:
            pd_pitch = pd_results[matched_index]['pitch']
            pd_yaw = pd_results[matched_index]['yaw']
            pd_roll = pd_results[matched_index]['roll']
            gt_data.append([gt_pitch, gt_yaw, gt_roll])
            pd_data.append([pd_pitch, pd_yaw, pd_roll])
 
            pd_results[matched_index]['gt_bbox'] = gt_bbox

            if abs(gt_yaw) > 90:
                gt_data_backward.append([gt_pitch, gt_yaw, gt_roll])
                pd_data_backward.append([pd_pitch, pd_yaw, pd_roll])
            else:
                gt_data_frontal.append([gt_pitch, gt_yaw, gt_roll])
                pd_data_frontal.append([pd_pitch, pd_yaw, pd_roll])
                
            if frontal_face:
                pd_results[matched_index]['gt_pitch'] = gt_pitch
                pd_results[matched_index]['gt_yaw'] = gt_yaw
                pd_results[matched_index]['gt_roll'] = gt_roll
                if abs(gt_yaw) < 90:
                    if str(image_id) not in appeared_image_id_list:
                        appeared_image_id_list.append(str(image_id))
                        gt_json_frontal_face['images'].append(gt_images_images_dict[str(image_id)])
                    gt_json_frontal_face['annotations'].append(gt_label_dict)
                    '''
                    This json file will be used for comparing with FAN, 3DDFA, FSA-Net and WHE-Net in narrow-range.
                    It will also be used for comparing with gt_json_frontal_face to calculate frontal bbox mAP.
                    '''
                    pd_json_frontal_face.append(pd_results[matched_index])  
                '''
                This json file will only be used for comparing with WHE-Net in full-range.
                '''
                pd_json_full_face.append(pd_results[matched_index])
                
         
    total_num = len(gt_labels_list)
    left_num = len(gt_data)
    
    if left_num == 0:
        return total_num, [30,60,30], 2, [30,60,30], 1, [30,60,30], 1
    
    if frontal_face:
        with open(pd_json_path[:-5]+"_pd_frontal.json", 'w') as f:
            json.dump(pd_json_frontal_face, f)
        with open(pd_json_path[:-5]+"_gt_frontal.json", 'w') as f:
            json.dump(gt_json_frontal_face, f)
        with open(pd_json_path[:-5]+"_pd_full.json", 'w') as f:
            json.dump(pd_json_full_face, f)
            
            
    error_list = np.abs(np.array(gt_data) - np.array(pd_data))
    error_list[:, 1] = np.min((error_list[:, 1], 360 - error_list[:, 1]), axis=0)  # yaw range is [-180,180]
    pose_matrix = np.mean(error_list, axis=0)
    
    left_num_b = len(gt_data_backward)
    if left_num_b != 0:
        error_list_backward = np.abs(np.array(gt_data_backward) - np.array(pd_data_backward))
        error_list_backward[:, 1] = np.min((error_list_backward[:, 1], 360 - error_list_backward[:, 1]), axis=0)  # yaw range is [-180,180]
        pose_matrix_b = np.mean(error_list_backward, axis=0)
    else:
        pose_matrix_b, left_num_b = [30,60,30], 1

    left_num_f = len(gt_data_frontal)
    if left_num_f != 0:
        error_list_frontal = np.abs(np.array(gt_data_frontal) - np.array(pd_data_frontal))
        error_list_frontal[:, 1] = np.min((error_list_frontal[:, 1], 360 - error_list_frontal[:, 1]), axis=0)  # yaw range is [-180,180]
        pose_matrix_f = np.mean(error_list_frontal, axis=0)
    else:
        pose_matrix_f, left_num_f = [30,60,30], 1
        
    return total_num, pose_matrix, left_num, pose_matrix_b, left_num_b, pose_matrix_f, left_num_f


def mean_absolute_error_calculate_single(gt_json_path, pd_json_path, frontal_face):
    bbox_iou_threshold = 0.5
    face_score_threshold = 0.01 # 0.3
    gt_data, pd_data = [], []  # shapes of both should be N*3
    gt_data_frontal, pd_data_frontal = [], []  # shapes of both should be N*3
    gt_data_backward, pd_data_backward = [], []  # shapes of both should be N*3

    gt_json = json.load(open(gt_json_path, "r"))
    pd_json = json.load(open(pd_json_path, "r"))
    
    gt_labels_list = gt_json['annotations']
    gt_images_labels_dict = sort_labels_by_image_id(gt_labels_list)
    pd_images_results_dict = sort_labels_by_image_id(pd_json)
    
    for image_id in gt_images_labels_dict.keys():
        label = gt_images_labels_dict[str(image_id)]  # only one valid face in each image
        if str(image_id) not in pd_images_results_dict:  # we may get zero face of one image
            continue
        results = pd_images_results_dict[str(image_id)]  # we may get more than one face of one image
        
        gt_bbox = label[0]['bbox']
        [gt_pitch, gt_yaw, gt_roll] = label[0]['euler_angles']
        
        max_iou, matched_index = 0, -1
        for i, pd_image_dict in enumerate(results):
            score = pd_image_dict['score']
            if score < face_score_threshold:
                continue
                
            pd_bbox = pd_image_dict['bbox']
            temp_iou = calculate_bbox_iou(pd_bbox, gt_bbox, format='xywh')
            if temp_iou > max_iou:
                max_iou = temp_iou
                matched_index = i

        if max_iou > bbox_iou_threshold:
            pd_pitch = results[matched_index]['pitch']
            pd_yaw = results[matched_index]['yaw']
            pd_roll = results[matched_index]['roll']
            gt_data.append([gt_pitch, gt_yaw, gt_roll])
            pd_data.append([pd_pitch, pd_yaw, pd_roll])  
                
            # if abs(gt_yaw) > 90:
            if abs(gt_yaw) > 99:
                gt_data_backward.append([gt_pitch, gt_yaw, gt_roll])
                pd_data_backward.append([pd_pitch, pd_yaw, pd_roll])
            else:
                gt_data_frontal.append([gt_pitch, gt_yaw, gt_roll])
                pd_data_frontal.append([pd_pitch, pd_yaw, pd_roll])
                
    total_num = len(gt_labels_list)
    left_num = len(gt_data)
    
    if left_num == 0:
        return total_num, [30,60,30], 2, [30,60,30], 1, [30,60,30], 1

    error_list = np.abs(np.array(gt_data) - np.array(pd_data))
    error_list[:, 1] = np.min((error_list[:, 1], 360 - error_list[:, 1]), axis=0)  # yaw range is [-180,180]
    pose_matrix = np.mean(error_list, axis=0)
    
    left_num_b = len(gt_data_backward)
    if left_num_b != 0:
        error_list_backward = np.abs(np.array(gt_data_backward) - np.array(pd_data_backward))
        error_list_backward[:, 1] = np.min((error_list_backward[:, 1], 360 - error_list_backward[:, 1]), axis=0)  # yaw range is [-180,180]
        pose_matrix_b = np.mean(error_list_backward, axis=0)
    else:
        pose_matrix_b, left_num_b = [30,60,30], 1

    left_num_f = len(gt_data_frontal)
    if left_num_f != 0:
        error_list_frontal = np.abs(np.array(gt_data_frontal) - np.array(pd_data_frontal))
        error_list_frontal[:, 1] = np.min((error_list_frontal[:, 1], 360 - error_list_frontal[:, 1]), axis=0)  # yaw range is [-180,180]
        pose_matrix_f = np.mean(error_list_frontal, axis=0)
    else:
        pose_matrix_f, left_num_f = [30,60,30], 1
        
    return total_num, pose_matrix, left_num, pose_matrix_b, left_num_b, pose_matrix_f, left_num_f  
