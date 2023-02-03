

'''
2023-01-03
The MOS annotation is not full enough !!! We cannot use it for our task

/datasdc/zhouhuayi/dataset/WiderFace/MOS-face_pose_label.txt  --> parse_mos_annotations() ...
WiderFace-MOS: original images-->12852, original face instances-->144335
Processing train-set ...
100%|███████████████████████████████████████████| 12880/12880 [01:35<00:00, 135.30it/s]
train: original images-->12880, left images-->8239, left face instances-->144335

Processing val-set ...
100%|███████████████████████████████████████████| 3226/3226 [00:00<00:00, 2429232.44it/s]
val: original images-->3226, left images-->0, left face instances-->0


2023-01-10
Loading frames paths... -->  /datasdc/zhouhuayi/dataset/WiderFace/img2pose_annotations/WIDER_val_annotations.txt
100%|████████████████████████████████████████| 3226/3226 [00:06<00:00, 469.27it/s]
WiderFace-img2pose (val): original images-->3226, original face instances-->39697
Processing val-set ...
100%|████████████████████████████████████████| 3226/3226 [00:25<00:00, 127.06it/s]
val: original images-->3226, left images-->3205, left face instances-->34294

Loading frames paths... -->  /datasdc/zhouhuayi/dataset/WiderFace/img2pose_annotations/WIDER_train_annotations.txt
100%|████████████████████████████████████████ 12880/12880 [00:27<00:00, 461.41it/s]
WiderFace-img2pose (train): original images-->12880, original face instances-->159393
Processing train-set ...
100%|████████████████████████████████████████| 12880/12880 [01:42<00:00, 126.21it/s]
train: original images-->12880, left images-->12874, left face instances-->138722
'''


import os
import cv2
import json
import copy
import shutil

from tqdm import tqdm
from scipy.spatial.transform import Rotation
from math import cos, sin, pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

coco_dict_template = {
    'info': {
        'description': 'Face 5 landmarks, Face bboxes and Euler angles of WiderFace Dataset',
        'url': 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html',
        'version': '1.0',
        'year': 2023,
        'contributor': 'Huayi Zhou',
        'date_created': '2023/01/10',
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

coco_style_dict_val = copy.deepcopy(coco_dict_template)
coco_style_dict_train = copy.deepcopy(coco_dict_template)

threed_5_points = np.array([
    [-0.30313677, -0.4008789 , -0.37974212],
    [ 0.32007796, -0.3944147 , -0.37752032],
    [ 0.01478795, -0.02910064, -0.7367204 ],
    [-0.23718083,  0.27580532, -0.39722505],
    [ 0.22684711,  0.28580052, -0.41195285]], dtype=np.float32)

threed_68_points = np.array([
    [-0.7424525 , -0.36621028,  0.4207102 ],
    [-0.740015  , -0.18364036,  0.56419116],
    [-0.6338956 ,  0.00513752,  0.1404103 ],
    [-0.59881735,  0.16176832, -0.01757937],
    [-0.545519  ,  0.335799  , -0.01980399],
    [-0.46692768,  0.4768002 , -0.1058567 ],
    [-0.37206355,  0.5835558 , -0.10784729],
    [-0.2199454 ,  0.6592754 , -0.35200933],
    [-0.01844856,  0.7018875 , -0.43120697],
    [ 0.18289383,  0.65876305, -0.4117388 ],
    [ 0.34133473,  0.5931947 , -0.22514082],
    [ 0.45350492,  0.50017023, -0.12005281],
    [ 0.55304134,  0.33640432, -0.01009766],
    [ 0.6050993 ,  0.16169539,  0.00174479],
    [ 0.60098666,  0.00495846,  0.21822056],
    [ 0.7229843 , -0.18304153,  0.5234964 ],
    [ 0.7263977 , -0.366935  ,  0.38820368],
    [-0.57411146, -0.5247366 , -0.16244768],
    [-0.4901746 , -0.6011213 , -0.3334854 ],
    [-0.37658313, -0.6216351 , -0.43373254],
    [-0.28903556, -0.6005562 , -0.48180974],
    [-0.19807056, -0.5750441 , -0.5064967 ],
    [ 0.15825556, -0.5988621 , -0.5168488 ],
    [ 0.24874154, -0.62013465, -0.49383065],
    [ 0.36312357, -0.621523  , -0.43854675],
    [ 0.47339717, -0.60108656, -0.34988633],
    [ 0.5571284 , -0.5475083 , -0.18696046],
    [-0.01819801, -0.3929245 , -0.5283855 ],
    [ 0.00495288, -0.26023945, -0.6294777 ],
    [-0.01810645, -0.15093385, -0.7110319 ],
    [-0.01807064, -0.06197068, -0.7462585 ],
    [-0.13046028,  0.02724737, -0.52054566],
    [-0.06473281,  0.050565  , -0.5579974 ],
    [ 0.00492744,  0.04997738, -0.5902355 ],
    [ 0.04803487,  0.05043504, -0.57315373],
    [ 0.11485971,  0.02745122, -0.532887  ],
    [-0.42329   , -0.35983515, -0.27480155],
    [-0.37829077, -0.42259988, -0.3739491 ],
    [-0.29025665, -0.42172942, -0.3799033 ],
    [-0.20009823, -0.39908078, -0.35613355],
    [-0.26669958, -0.35450596, -0.36584428],
    [-0.37644294, -0.35356686, -0.3440864 ],
    [ 0.18348028, -0.39950916, -0.35511076],
    [ 0.250092  , -0.4218903 , -0.37414902],
    [ 0.34109643, -0.42234853, -0.3760238 ],
    [ 0.4082288 , -0.3986789 , -0.3338025 ],
    [ 0.34100366, -0.35501942, -0.3625706 ],
    [ 0.24881038, -0.3763026 , -0.3651737 ],
    [-0.23737977,  0.26947072, -0.40859538],
    [-0.17359589,  0.2256999 , -0.502612  ],
    [-0.06435671,  0.1822612 , -0.5702637 ],
    [ 0.00494695,  0.2052055 , -0.57842386],
    [ 0.04794669,  0.18260588, -0.5738539 ],
    [ 0.15626651,  0.22453448, -0.51302433],
    [ 0.2441238 ,  0.2696989 , -0.40121093],
    [ 0.1572282 ,  0.3153274 , -0.49054393],
    [ 0.07134633,  0.33926618, -0.54565275],
    [ 0.00498485,  0.3397523 , -0.5556763 ],
    [-0.08456777,  0.33906335, -0.5393014 ],
    [-0.15052275,  0.3151444 , -0.49258518],
    [-0.23737977,  0.26947072, -0.40859538],
    [-0.08448927,  0.24931723, -0.5288047 ],
    [ 0.00496338,  0.24893254, -0.55139536],
    [ 0.07113311,  0.24894084, -0.5354389 ],
    [ 0.22453459,  0.2697715 , -0.41060132],
    [ 0.07113311,  0.24894084, -0.5354389 ],
    [ 0.00496338,  0.24893254, -0.55139536],
    [-0.06450798,  0.24891317, -0.53638107]], dtype=np.float32)


############################################################################################
'''
refer code:
https://github.com/vitoralbiero/img2pose#prepare-wider-face-dataset
https://github.com/vitoralbiero/img2pose/convert_json_list_to_lmdb.py
https://github.com/vitoralbiero/img2pose/utils/json_loader.py
https://github.com/vitoralbiero/img2pose/utils/pose_operations.py
'''
def parse_img2pose_annotations(face_pose_label_img2pose, anno_text_name):
    
    annos_dict_list = {}
    instances_cnt = 0
    
    json_list = os.path.join(face_pose_label_img2pose, anno_text_name)
    image_anno_paths = pd.read_csv(json_list, delimiter=" ", header=None)
    image_anno_paths = np.asarray(image_anno_paths).squeeze()
    
    print("Loading frames paths... --> ", json_list)
    for image_anno_path in tqdm(image_anno_paths):
        
        image_anno_path_real = image_anno_path.replace("annotations", face_pose_label)
        with open(image_anno_path_real) as f:
            image_json = json.load(f)
            
        img_path = image_json["image_path"]  # e.g., "0--Parade/0_Parade_marchingband_1_20.jpg"
        bboxes = image_json["bboxes"]
        landmarks = image_json["landmarks"]
        
        annos_dict_list[img_path] = []
        
        for bbox, landmark in zip(bboxes, landmarks):
            bbox = np.asarray(bbox)[:4].astype(float)
            landmark = np.asarray(landmark)[:, :2].astype(float)

            # remove samples that do not have height ot width or are negative
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue

            new_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # [x, y, w, h]
                
            temp_img_dict = {}
            if -1 in landmark:
                temp_img_dict["euler_angles"] = [999, 999, 999]  # head pose is not labeled
                
                temp_img_dict["bbox"] = new_bbox  # 1 x 4, in [x, y, w, h] format
                temp_img_dict["landmark"] = landmark.tolist()  # n x 2, n = 5 or 68
            else:
                landmark[:, 0] -= new_bbox[0]
                landmark[:, 1] -= new_bbox[1]
            
                w, h = int(new_bbox[2]), int(new_bbox[3])
                bbox_intrinsics = np.array([
                    [w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])

                if len(landmark) == 5:
                    P, pose = get_pose(threed_5_points, landmark, bbox_intrinsics)
                else:
                    P, pose = get_pose(threed_68_points, landmark, bbox_intrinsics)

                rotvec = pose[:3]  # 1 x 6 --> 1 x 3
                rot_mat_1 = Rotation.from_rotvec(rotvec).as_matrix()  # rot_vector --> rot_matrix
                rot_mat_2 = np.transpose(rot_mat_1)
                angle = Rotation.from_matrix(rot_mat_2).as_euler('xyz', degrees=True)  # rot_matrix --> euler  
                # np.array([angle[0], -angle[1], -angle[2]]) --> Pitch, Yaw, Roll
                Pitch, Yaw, Roll = float(angle[0]), float(-angle[1]), float(-angle[2])
                Pitch = -90 if (Pitch < -90) else 90 if (Pitch > 90) else Pitch  # pitch in [-90, 90]
                Roll = -90 if (Roll < -90) else 90 if (Roll > 90) else Roll  # roll in [-90, 90]
                temp_img_dict["euler_angles"] = [Pitch, Yaw, Roll]
            
                temp_img_dict["bbox"] = new_bbox  # 1 x 4, in [x, y, w, h] format
                temp_img_dict["landmark"] = landmark.tolist()  # n x 2, n = 5 or 68
            
            # each image contains multiple annotation dict
            annos_dict_list[img_path].append(temp_img_dict)
            instances_cnt += 1
            
    return annos_dict_list, instances_cnt


def get_pose(vertices, twod_landmarks, camera_intrinsics, initial_pose=None):
    threed_landmarks = vertices
    twod_landmarks = np.asarray(twod_landmarks).astype("float32")

    # if initial_pose is provided, use it as a guess to solve new pose
    if initial_pose is not None:
        initial_pose = np.asarray(initial_pose)
        retval, rvecs, tvecs = cv2.solvePnP(
            threed_landmarks,
            twod_landmarks,
            camera_intrinsics,
            None,
            rvec=initial_pose[:3],
            tvec=initial_pose[3:],
            flags=cv2.SOLVEPNP_EPNP,
            useExtrinsicGuess=True,
        )
    else:
        retval, rvecs, tvecs = cv2.solvePnP(
            threed_landmarks,
            twod_landmarks,
            camera_intrinsics,
            None,
            flags=cv2.SOLVEPNP_EPNP,
        )

    rotation_mat = np.zeros(shape=(3, 3))
    R = cv2.Rodrigues(rvecs, rotation_mat)[0]

    RT = np.column_stack((R, tvecs))
    P = np.matmul(camera_intrinsics, RT)
    dof = np.append(rvecs, tvecs)

    return P, dof


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
    
    
def convert_to_coco_style(source_img, target_img, annos_dict_list, dataset_type="train"):
    
    debug = True
    # debug = False
    
    print("Processing %s-set ..."%(dataset_type))
    
    if os.path.exists(os.path.join(target_img, dataset_type)):
        shutil.rmtree(os.path.join(target_img, dataset_type))
    os.mkdir(os.path.join(target_img, dataset_type))
    
    for image_index, (folder_img_name, annos_dict) in enumerate(tqdm(annos_dict_list.items())):
        
        if dataset_type == "train":
            image_id = 100000 + image_index
        if dataset_type == "val":
            image_id = 200000 + image_index
        new_file_name = str(image_id)+".jpg"
        
        source_img_path = os.path.join(source_img, dataset_type, folder_img_name)
        target_img_path = os.path.join(target_img, dataset_type, new_file_name)
        folder, imgname = folder_img_name.split("/")  
        
        img = cv2.imread(source_img_path)
        img_h, img_w, img_c = img.shape

        '''coco_style_sample'''
        image_dict = {
            'file_name': new_file_name,
            'height': img_h,
            'width': img_w,
            'id': image_id,
            'img_ori_name': imgname,
        }
        
        temp_annotations_list = []
        for ind, labels in enumerate(annos_dict):
            if labels["euler_angles"][0] == 999:  # head pose is not labeled
                continue
                
            # we may need to enlarge the original face/head bboxes for our DirectMHP use
                
            temp_annotation = {
                'landmark': labels["landmark"],  # n x 2, n = 5 or 68
                'bbox': labels["bbox"],  # face bbox, [x, y, w, h]
                'euler_angles': labels["euler_angles"],  # format [pitch, yaw, roll] in degree
                'image_id': image_id,  # int
                'id': image_id * 1000 + ind,  # we support that no image has more than 1000 faces/poses
                'category_id': 1,
                'iscrowd': 0,
                'segmentation': [],  # This script is not for segmentation
                'area': round(labels["bbox"][-1] * labels["bbox"][-2], 4)
            }
            temp_annotations_list.append(temp_annotation)

        if len(temp_annotations_list) != 0:
            if dataset_type == "train":
                coco_style_dict_train['images'].append(image_dict)
                coco_style_dict_train['annotations'] += temp_annotations_list
            if dataset_type == "val":
                coco_style_dict_val['images'].append(image_dict)
                coco_style_dict_val['annotations'] += temp_annotations_list
            shutil.copy(source_img_path, target_img_path)
        else:
            continue  # this image has no annotations left


        if image_index < 20 and debug == True:
            for ind, labels in enumerate(annos_dict):
                [x, y, w, h] = labels["bbox"]
                cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (255,255,255), 2)  # plot bbox
                [pitch, yaw, roll] = labels["euler_angles"]
                if pitch != 999:  # plot head pose
                    img = draw_axis(img, yaw, pitch, roll, tdx=x+w/2, tdy=y+h/2, size=100)
            save_img_path = "./tmp/" + dataset_type + "_" + imgname
            cv2.imwrite(save_img_path, img)


    all_folders = os.listdir(os.path.join(source_img, dataset_type))
    all_imgnames = []
    for folder in all_folders:
        imgnames = os.listdir(os.path.join(source_img, dataset_type, folder))
        imgnames = [folder+"/"+temp for temp in imgnames]
        all_imgnames += imgnames

    return len(all_imgnames)

if __name__ == "__main__":

    image_file_origin = "/datasdc/zhouhuayi/dataset/WiderFace/images_original"
    image_file_resort = "/datasdc/zhouhuayi/dataset/WiderFace/images"
    
    anno_save_folder_train = "/datasdc/zhouhuayi/dataset/WiderFace/annotations/coco_style_img2pose_train.json"
    anno_save_folder_val = "/datasdc/zhouhuayi/dataset/WiderFace/annotations/coco_style_img2pose_val.json"
    
    face_pose_label = "/datasdc/zhouhuayi/dataset/WiderFace/img2pose_annotations" 
    euler_angles_stat = [[],[],[]]  # pitch, yaw, roll


    anno_val_name = "WIDER_val_annotations.txt"
    annos_dict_list_val, instances_cnt = parse_img2pose_annotations(face_pose_label, anno_val_name)
    print("WiderFace-img2pose (val): original images-->%d, original face instances-->%d"%(
        len(annos_dict_list_val), instances_cnt))
        
    total_images_origin = convert_to_coco_style(
        image_file_origin, image_file_resort, annos_dict_list_val, dataset_type="val")
    with open(anno_save_folder_val, "w") as dst_ann_file:
        json.dump(coco_style_dict_val, dst_ann_file)
    print("val: original images-->%d, left images-->%d, left face instances-->%d\n"%(
        total_images_origin, len(coco_style_dict_val['images']), len(coco_style_dict_val['annotations'])))
    for labels_dict in coco_style_dict_val['annotations']:
        [pitch, yaw, roll] = labels_dict['euler_angles']
        euler_angles_stat[0].append(pitch)
        euler_angles_stat[1].append(yaw)
        euler_angles_stat[2].append(roll) 


    anno_train_name = "WIDER_train_annotations.txt"
    annos_dict_list_train, instances_cnt = parse_img2pose_annotations(face_pose_label, anno_train_name)
    print("WiderFace-img2pose (train): original images-->%d, original face instances-->%d"%(
        len(annos_dict_list_train), instances_cnt))
        
    total_images_origin = convert_to_coco_style(
        image_file_origin, image_file_resort, annos_dict_list_train, dataset_type="train")
    with open(anno_save_folder_train, "w") as dst_ann_file:
        json.dump(coco_style_dict_train, dst_ann_file)
    print("train: original images-->%d, left images-->%d, left face instances-->%d\n"%(
        total_images_origin, len(coco_style_dict_train['images']), len(coco_style_dict_train['annotations'])))
    for labels_dict in coco_style_dict_train['annotations']:
        [pitch, yaw, roll] = labels_dict['euler_angles']
        euler_angles_stat[0].append(pitch)
        euler_angles_stat[1].append(yaw)
        euler_angles_stat[2].append(roll)



    '''WiderFace-img2pose Euler Angels Stat'''
    interval = 10  # 10 or 15 is better
    bins = 200 // interval  # 360 // interval
    density = False  # True or False, density=False would make counts
    colors = ['r', 'g', 'b']
    labels = ["Pitch", "Yaw", "Roll"]
    plt.hist(euler_angles_stat, bins=bins, alpha=0.7, density=density, histtype='bar', label=labels, color=colors)
    plt.legend(prop ={'size': 10})
    # plt.xlim(-180, 180)
    plt.xticks(range(-100,101,interval))  # plt.xticks(range(-180,181,interval))
    if density: plt.ylabel('Percentage')
    else: plt.ylabel('Counts')
    plt.xlabel('Degree')
    plt.show()
    
    
