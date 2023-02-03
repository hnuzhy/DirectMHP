
import os
import json
import copy
import numpy as np
from tqdm import tqdm

import shutil
import matplotlib.pyplot as plt

############################################################################################

# Face keypoint orders follow Openpose keypoint output
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
# Face outline points (0-16) are unstable
face_edges = np.array([ 
    # [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[11,12],[12,13],[14,15],[15,16], #outline (ignored)
    [17,18],[18,19],[19,20],[20,21], #right eyebrow
    [22,23],[23,24],[24,25],[25,26], #left eyebrow
    [27,28],[28,29],[29,30],   #nose upper part
    [31,32],[32,33],[33,34],[34,35], #nose lower part
    [36,37],[37,38],[38,39],[39,40],[40,41],[41,36], #right eye
    [42,43],[43,44],[44,45],[45,46],[46,47],[47,42], #left eye
    [48,49],[49,50],[50,51],[51,52],[52,53],[53,54],[54,55],[55,56],[56,57],[57,58],[58,59],[59,48], #Lip outline
    [60,61],[61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,60] #Lip inner line 
    ])

coco_dict_template = {
    'info': {
        'description': 'Face landmarks and Euler angles of CMU Panoptic Studio Dataset',
        'url': 'http://domedb.perception.cs.cmu.edu/',
        'version': '1.0',
        'year': 2022,
        'contributor': 'Huayi Zhou',
        'date_created': '2022/02/17',
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
        'name': 'person',
        'face_edges': face_edges.tolist()
    }]
}

############################################################################################

def sort_labels_by_image_id(labels_list):
    images_labels_dict = {}
    for i, labels_dict in enumerate(labels_list):
        image_id = str(labels_dict['image_id'])
        if 'head_bbox' in labels_dict:
            labels_dict['bbox'] = labels_dict['head_bbox']  # please use the default 'bbox' as key in cocoapi
            del labels_dict['head_bbox']
        if 'area' not in labels_dict:  # generate standard COCO style json file
            labels_dict['segmentation'] = []  # This script is not for segmentation
            labels_dict['area'] = round(labels_dict['bbox'][-1] * labels_dict['bbox'][-2], 4)
        if image_id in images_labels_dict.keys():
            images_labels_dict[image_id].append(labels_dict)
        else:
            images_labels_dict[image_id] = [labels_dict]
    return images_labels_dict


if __name__ == "__main__":
    
    sampled_anno_path = "./HPE/annotations/coco_style_sample.json"
    sampled_train_path = "./HPE/annotations/coco_style_sampled_train.json"
    sampled_val_path = "./HPE/annotations/coco_style_sampled_val.json"
    
    image_root_path = "./HPE/images_sampled"
    
    image_dst_path = "./HPE/images"
    if os.path.exists(image_dst_path):
        shutil.rmtree(image_dst_path)
    os.mkdir(image_dst_path)
    os.mkdir(os.path.join(image_dst_path, "train"))
    os.mkdir(os.path.join(image_dst_path, "val"))
    
    
    '''[start] do not change'''
    seq_names = ["171204_pose3", "171026_pose3", "170221_haggling_b3", "170221_haggling_m3", "170224_haggling_a3", "170228_haggling_b1", "170404_haggling_a1", "170407_haggling_a2", "170407_haggling_b2", "171026_cello3", "161029_piano4", "160422_ultimatum1", "160224_haggling1", "170307_dance5", "160906_ian1", "170915_office1", "160906_pizza1"]  # 17 names
    
    seq_names_train = ["171204_pose3", "161029_piano4", "160422_ultimatum1", "170307_dance5", "160906_pizza1", "170221_haggling_b3", "170224_haggling_a3", "170404_haggling_a1", "170407_haggling_b2"]  # 9 names, person: 1+1+7+1+5+3+3+3+3
    seq_names_val = ["171026_pose3", "171026_cello3", "160224_haggling1", "160906_ian1", "170915_office1", "170221_haggling_m3", "170228_haggling_b1", "170407_haggling_a2"]  # 8 names, person: 1+1+3+2+1+3+3+3
    

    train_seq_num_list, val_seq_num_list = [], []
    for seq_num, seq_name in enumerate(seq_names):
        if seq_name in seq_names_train: train_seq_num_list.append(seq_num)
        if seq_name in seq_names_val: val_seq_num_list.append(seq_num)

    with open(sampled_anno_path, "r") as json_file:
        annos_dict = json.load(json_file)
    images_list = annos_dict['images']
    labels_list = annos_dict['annotations']
    images_labels_dict = sort_labels_by_image_id(labels_list)

    coco_dict_train = copy.deepcopy(coco_dict_template)
    coco_dict_val = copy.deepcopy(coco_dict_template)
    
    person_instances_stat = {}
    euler_angles_stat = [[],[],[]]  # pitch, yaw, roll

    for image_dict in tqdm(images_list):
        image_id = image_dict['id']
        seq_num = (image_id - 10000000000) // 100000000 - 1
        if seq_num in train_seq_num_list: target_type = "train"
        if seq_num in val_seq_num_list: target_type = "val"
        
        labels_list = images_labels_dict[str(image_id)]
        anno_nums = len(labels_list)

        image_dict['seq'] = seq_names[seq_num]
        
        src_image_path = os.path.join(image_root_path, image_dict['file_name'])
        dst_image_path = os.path.join(image_dst_path, target_type, image_dict['file_name'])
        if os.path.exists(src_image_path):
            shutil.move(src_image_path, dst_image_path)

        if target_type == "train":
            coco_dict_train['images'].append(image_dict)
            coco_dict_train['annotations'] += labels_list
            if str(anno_nums) not in person_instances_stat:
                person_instances_stat[str(anno_nums)] = [1,0]  # [1, 0] for [train, val]
            else:
                person_instances_stat[str(anno_nums)][0] += 1
        if target_type == "val":
            coco_dict_val['images'].append(image_dict)
            coco_dict_val['annotations'] += labels_list
            if str(anno_nums) not in person_instances_stat:
                person_instances_stat[str(anno_nums)] = [0,1]  # [0, 1] for [train, val]
            else:
                person_instances_stat[str(anno_nums)][1] += 1
        
        for labels in labels_list:
            [pitch, yaw, roll] = labels['euler_angles']
            euler_angles_stat[0].append(pitch)
            euler_angles_stat[1].append(yaw)
            euler_angles_stat[2].append(roll)
            
    '''[end] do not change'''
    
    print("\nperson_instances_stat:", person_instances_stat)
    image_cnt, person_cnt = [0,0], [0,0]
    for key, value in person_instances_stat.items():
        image_cnt[0], image_cnt[1] = image_cnt[0] + value[0], image_cnt[1] + value[1]
        person_cnt[0], person_cnt[1] = person_cnt[0] + int(key)*value[0], person_cnt[1] + int(key)*value[1]
        print("Images number containing [%s] persons: %d, \ttrain/val = %d/%d"%(key, sum(value), value[0], value[1]))
    print("Perosn instances per image: %.4f, \ttrain/val = %.4f/%.4f"%(
        sum(person_cnt)/sum(image_cnt), person_cnt[0]/image_cnt[0], person_cnt[1]/image_cnt[1]))

    print("\ntrain: images --> %d, head instances --> %d"%(len(coco_dict_train['images']), len(coco_dict_train['annotations'])))  
    with open(sampled_train_path, "w") as json_file:
        json.dump(coco_dict_train, json_file)
    print("val: images --> %d, head instances --> %d"%(len(coco_dict_val['images']), len(coco_dict_val['annotations'])))
    with open(sampled_val_path, "w") as json_file:
        json.dump(coco_dict_val, json_file)
    
    '''CMUPanoptic Euler Angels Stat'''
    interval = 10  # 10 or 15 is better
    bins = 360 // interval
    density = False  # True or False, density=False would make counts
    colors = ['r', 'g', 'b']
    labels = ["Pitch", "Yaw", "Roll"]
    plt.hist(euler_angles_stat, bins=bins, alpha=0.7, density=density, histtype='bar', label=labels, color=colors)
    plt.legend(prop ={'size': 10})
    # plt.xlim(-180, 180)
    plt.xticks(range(-180,181,interval))
    if density: plt.ylabel('Percentage')
    else: plt.ylabel('Counts')
    plt.xlabel('Degree')
    plt.show()


    '''final results
    100%|███████████████████████████████████████████████████████████████████████████████████████| 31934/31934 [00:40<00:00, 794.51it/s]

    person_instances_stat: {'1': [7416, 7291], '2': [1313, 1328], '3': [4937, 7597], '4': [479, 0], '5': [567, 0], '7': [85, 0], '6': [921, 0]}
    Images number containing [1] persons: 14707,    train/val = 7416/7291
    Images number containing [2] persons: 2641,     train/val = 1313/1328
    Images number containing [3] persons: 12534,    train/val = 4937/7597
    Images number containing [4] persons: 479,      train/val = 479/0
    Images number containing [5] persons: 567,      train/val = 567/0
    Images number containing [7] persons: 85,       train/val = 85/0
    Images number containing [6] persons: 921,      train/val = 921/0
    Perosn instances per image: 2.1439,     train/val = 2.2729/2.0189

    train: images --> 15718, head instances --> 35725
    val: images --> 16216, head instances --> 32738
    '''