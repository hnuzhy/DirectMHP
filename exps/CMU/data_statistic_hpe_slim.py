
import os
import json
import cv2
import shutil
import numpy as np
from tqdm import tqdm

from hpe_utils import projectPoints
from hpe_utils import align_3d_head
from hpe_utils import reference_head
from hpe_utils import get_sphere
from hpe_utils import select_euler
from hpe_utils import inverse_rotate_zyx

from hpe_utils import draw_axis, plot_pose_cube, plot_cuboid_Zaxis_by_euler_angles

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

coco_style_hpe_dict = {
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

# init transform params

E_ref = np.mat([[1, 0, 0, 0.],
            [0, -1, 0, 0],
            [0, 0, -1, 50],
            [0, 0, 0,  1]])

model_points, _ = reference_head(scale=1, pyr=(0., 0., 0.))
model_points_3D = np.ones((4, 58), dtype=np.float32)
model_points_3D[0:3] = model_points

kp_idx = np.asarray([17, 21, 26, 22, 45, 42, 39, 36, 35, 31, 54, 48, 57, 8])  # 14 indexs of refered points in CMUPanoptic
kp_idx_model = np.asarray([38, 34, 33, 29, 13, 17, 25, 21, 54, 50, 43, 39, 45, 6])  # 14 indexs of refered points in FaceModel

kp_idx_10       = np.asarray([17, 26, 42, 39, 35, 31, 54, 48, 57, 8])  # 10
kp_idx_model_10 = np.asarray([38, 33, 17, 25, 54, 50, 43, 39, 45, 6])  # 10

kp_idx_18       = np.asarray([17, 19, 21, 26, 24, 22, 45, 42, 39, 36, 35, 33, 31, 54, 48, 51, 57, 8])  # 18
kp_idx_model_18 = np.asarray([38, 36, 34, 33, 31, 29, 13, 17, 25, 21, 54, 52, 50, 43, 39, 41, 45, 6])  # 18

sphere = []
for theta in range(0, 360, 10):
    for phi in range(0, 180, 10):
        # sphere.append(get_sphere(theta, phi, 22))  # default radius is 22
        sphere.append(get_sphere(theta, phi, 18))
sphere = np.asarray(sphere)
sphere = sphere + [0, 5, -5]
sphere = sphere.T

############################################################################################


def parsing_camera_calibration_params(hd_camera_path):
    with open(hd_camera_path) as cfile:
        calib = json.load(cfile)

    # Cameras are identified by a tuple of (panel#,node#)
    cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}

    # Convert data into numpy arrays for convenience
    for k,cam in cameras.items():    
        cam['K'] = np.matrix(cam['K'])
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['R'] = np.matrix(cam['R'])
        cam['t'] = np.array(cam['t']).reshape((3,1))
        
    return cameras
    

def auto_labels_generating(cam, fframe_dict, frame):

    valid_bbox_euler_list = []
    
    img_h, img_w = frame.shape[0], frame.shape[1]
    
    cam['K'] = np.mat(cam['K'])
    cam['distCoef'] = np.array(cam['distCoef'])
    cam['R'] = np.mat(cam['R'])
    cam['t'] = np.array(cam['t']).reshape((3, 1))
    
    lost_faces = 0
    
    for face in fframe_dict['people']:
        # 3D Face has 70 3D joints, stored as an array [x1,y1,z1,x2,y2,z2,...]
        face3d = np.array(face['face70']['landmarks']).reshape((-1, 3)).transpose()
        face_conf = np.asarray(face['face70']['averageScore'])
        clean_match = (face_conf[kp_idx] > 0.1)  # only pick points confidence higher than 0.1
        kp_idx_clean = kp_idx[clean_match]
        kp_idx_model_clean = kp_idx_model[clean_match]
        
        clean_match_10 = (face_conf[kp_idx_10] > 0.1)  # only pick points confidence higher than 0.1
        kp_idx_clean_10 = kp_idx_10[clean_match_10]
        kp_idx_model_clean_10 = kp_idx_model_10[clean_match_10]

        clean_match_18 = (face_conf[kp_idx_18] > 0.1)  # only pick points confidence higher than 0.1
        kp_idx_clean_18 = kp_idx_18[clean_match_18]
        kp_idx_model_clean_18 = kp_idx_model_18[clean_match_18]

        if(len(kp_idx_clean)>6):
            rotation, translation, error, scale = align_3d_head(
                np.mat(model_points_3D[0:3, kp_idx_model_clean]), np.mat(face3d[:, kp_idx_clean]))
            
            sphere_new = scale * rotation @ (sphere) + translation
            pt_helmet = projectPoints(sphere_new, cam['K'], cam['R'], cam['t'], cam['distCoef'])
                
            temp = np.zeros((4, 4))
            temp[0:3, 0:3] = rotation
            temp[0:3, 3:4] = translation
            temp[3, 3] = 1
            E_virt = np.linalg.inv(temp @ np.linalg.inv(E_ref))
            
            E_real = np.zeros((4, 4))
            E_real[0:3, 0:3] = cam['R']
            E_real[0:3, 3:4] = cam['t']
            E_real[3, 3] = 1

            compound = E_real @ np.linalg.inv(E_virt)
            status, [pitch, yaw, roll] = select_euler(np.rad2deg(inverse_rotate_zyx(compound)))
            yaw = -yaw
            roll = -roll
            
            
            rotation_10, translation_10, _, _ = align_3d_head(
                np.mat(model_points_3D[0:3, kp_idx_model_clean_10]), np.mat(face3d[:, kp_idx_clean_10]))
            temp[0:3, 0:3] = rotation_10
            temp[0:3, 3:4] = translation_10
            E_virt = np.linalg.inv(temp @ np.linalg.inv(E_ref))
            compound = E_real @ np.linalg.inv(E_virt)
            status_10, [pitch_10, yaw_10, roll_10] = select_euler(np.rad2deg(inverse_rotate_zyx(compound)))
            yaw_10 = -yaw_10
            roll_10 = -roll_10
            
            rotation_18, translation_18, _, _ = align_3d_head(
                np.mat(model_points_3D[0:3, kp_idx_model_clean_18]), np.mat(face3d[:, kp_idx_clean_18]))
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
                    face2d = projectPoints(face3d, cam['K'], cam['R'], cam['t'], cam['distCoef'])
                    head_bbox = [x_min, y_min, w, h]  # format [x,y,w,h]
                    euler_angles = [pitch, yaw, roll]  # represented by degree
                    valid_bbox_euler_list.append({
                        "face2d_pts": [list(face2d[0]), list(face2d[1])], 
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
        else:
            # print("The face in this frame is not having enough valid 3D landmarks...")
            lost_faces += 1
            continue
    return valid_bbox_euler_list, lost_faces
    

'''Sampling frames by opencv2, frame_ids and labels could be aligned successfully'''
def process_and_extract_frames_face3d_cv2read(raw_data_path, sampled_result_path, seq_names, skip_frames, debug=True):
    # if os.path.exists(os.path.join(sampled_result_path, "images_sampled")):
        # shutil.rmtree(os.path.join(sampled_result_path, "images_sampled"))
    # os.mkdir(os.path.join(sampled_result_path, "images_sampled"))
    
    # if os.path.exists(os.path.join(sampled_result_path, "annotations")):
        # shutil.rmtree(os.path.join(sampled_result_path, "annotations"))
    # os.mkdir(os.path.join(sampled_result_path, "annotations"))

    img_w, img_h = 1920, 1080  # this is the default resolution of hd videos
    
    if debug:
        # seq_names = ["170221_haggling_b3", "171026_cello3"]  # for testing
        seq_names = ["171204_pose3", "171026_pose3"]
        
    for id, seq_name in enumerate(tqdm(seq_names)):
        hd_camera_path = os.path.join(raw_data_path, seq_name, "calibration_%s.json"%(seq_name))
        hd_face3d_path = os.path.join(raw_data_path, seq_name, "hdFace3d")
        
        cameras = parsing_camera_calibration_params(hd_camera_path)
        
        for vi in range(31):  # index 0~30, totally 31 hd videos
            cam = cameras[(0,vi)]
            
            hd_video_path = os.path.join(raw_data_path, seq_name, "hdVideos", 'hd_00_{0:02d}.mp4'.format(vi))
            if not os.path.exists(hd_video_path):
                continue  # This hd_video_path may be not existed, skip it
            
            capture = cv2.VideoCapture(hd_video_path)
            video_fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_seconds = int(frame_count / video_fps)
            video_duration = str(video_seconds // 60).zfill(2) + ":" + str(video_seconds % 60).zfill(2)
            print(id, vi, seq_name, "fps:", video_fps, "frame_cnt:", frame_count, "duration:", video_duration)
            
            if video_seconds > 5*60:  # more than 5 minutes
                interval = skip_frames*3
            else:
                interval = skip_frames
            
            for frame_id in range(frame_count):
                if debug:  # for faster testing
                    if frame_id > 900: break
                    # if frame_id > 300: break
            
                # ret, frame = capture.read()
                if frame_id % interval == 0:
                    real_frame_id = frame_id 
                else:
                    continue  # This frame is not chosen, skip it
                    
                image_id = 10000000000 + (id+1)*100000000 + real_frame_id * 100 + vi # 1xxzzzzzzvv
                dst_img_path_train = os.path.join(sampled_result_path, "images/train/", str(image_id)+".jpg")
                dst_img_path_val = os.path.join(sampled_result_path, "images/val/", str(image_id)+".jpg")
                if (not os.path.exists(dst_img_path_train)) and (not os.path.exists(dst_img_path_val)):
                    continue   # this frame may be damaged, skip it
                else:
                    if os.path.exists(dst_img_path_train):
                        frame = cv2.imread(dst_img_path_train)
                    if os.path.exists(dst_img_path_val):
                        frame = cv2.imread(dst_img_path_val)
                        
                face_json_fname = os.path.join(hd_face3d_path, 'faceRecon3D_hd{0:08d}.json'.format(real_frame_id))
                if not os.path.exists(face_json_fname):
                    continue  # This real_frame_id has no json annotation, skip it
                    
                with open(face_json_fname) as dfile:  # Load the json file with this frame's face
                    fframe = json.load(dfile)
                valid_bbox_euler_list, lost_faces = auto_labels_generating(cam, fframe, frame)
                if len(valid_bbox_euler_list) == 0:
                    continue  # This real_frame_id with NULL valid annotation, skip it
                if lost_faces != 0:
                    continue  # This frame has some un-labeled faces (with weak landmarks annotation), skip it
                
                
                if debug:  # visualization for testing
                    cv2.putText(frame, "lost_faces:"+str(lost_faces), (10,40), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), thickness=2)
                    for bbox_euler in valid_bbox_euler_list:
                        head_bbox, euler_angles = bbox_euler["head_bbox"], bbox_euler["euler_angles"]
                        [x_min, y_min, w, h] = head_bbox
                        [pitch, yaw, roll] = euler_angles
                        frame = cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_min+w), int(y_min+h)), (255,255,255), 2)
                        frame = plot_cuboid_Zaxis_by_euler_angles(frame, yaw, pitch, roll, 
                            tdx=x_min+w/2, tdy=y_min+h/2, limited=True)
                        for fai, fbi in face_edges:
                            ptxa, ptya = bbox_euler["face2d_pts"][0][fai], bbox_euler["face2d_pts"][1][fai]
                            ptxb, ptyb = bbox_euler["face2d_pts"][0][fbi], bbox_euler["face2d_pts"][1][fbi]
                            cv2.line(frame, (int(ptxa), int(ptya)), (int(ptxb), int(ptyb)), (255,255,0), 2)
                            
                    dst_img_path = os.path.join(sampled_result_path, "images_sampled", str(image_id)+".jpg")
                    cv2.imwrite(dst_img_path, frame)
                    
                # image_id = 10000000000 + (id+1)*100000000 + real_frame_id * 100 + vi # 1xxzzzzzzvv
                # dst_img_path = os.path.join(sampled_result_path, "images_sampled", str(image_id)+".jpg")
                # cv2.imwrite(dst_img_path, frame)
                # if os.path.getsize(dst_img_path) < 100*1024:  # <100KB
                    # os.remove(dst_img_path)
                    # continue  # this frame may be damaged, skip it

                
                '''coco_style_sample'''
                temp_image = {'file_name': str(image_id)+".jpg", 'height': img_h, 'width': img_w, 'id': image_id}
                temp_annotations_list = []
                for index, labels in enumerate(valid_bbox_euler_list):
                    temp_annotation = {
                        # 'face2d_pts': labels["face2d_pts"],
                        'bbox': labels["head_bbox"],  # please use the default 'bbox' as key in cocoapi
                        'euler_angles': labels["euler_angles"],  # with 14 reference points
                        'euler_angles_10': labels["euler_angles_10"],  # with 10 reference points
                        'euler_angles_18': labels["euler_angles_18"],  # with 18 reference points
                        'image_id': image_id,
                        'id': image_id * 100 + index,  # we support that no image has more than 100 persons/poses
                        'category_id': 1,
                        'iscrowd': 0,
                        # 'segmentation': [],  # This script is not for segmentation
                        # 'area': round(labels["head_bbox"][-1] * labels["head_bbox"][-2], 4)
                    }
                    temp_annotations_list.append(temp_annotation)
                
                coco_style_hpe_dict['images'].append(temp_image)
                coco_style_hpe_dict['annotations'] += temp_annotations_list
                
            capture.release()

        print(id, "\t finish one seq named:", seq_name)
        
    dst_ann_path = os.path.join(sampled_result_path, "annotations", "coco_style_sample_slim.json")
    with open(dst_ann_path, "w") as dst_ann_file:
        json.dump(coco_style_hpe_dict, dst_ann_file)
    

if __name__ == "__main__":

    raw_data_path = "./panoptic-toolbox_HPE/"
    sampled_result_path = "./HPE/"
    
    seq_names = ["171204_pose3", "171026_pose3", "170221_haggling_b3", "170221_haggling_m3", "170224_haggling_a3", "170228_haggling_b1", "170404_haggling_a1", "170407_haggling_a2", "170407_haggling_b2", "171026_cello3", "161029_piano4", "160422_ultimatum1", "160224_haggling1", "170307_dance5", "160906_ian1", "170915_office1", "160906_pizza1"]  # 17 names

    '''
    Sampled HD video frames every skip_frames (e.g. FPS=30) frames.
    We totally have about 7020 seconds and 31 hd views. 
    Set 60 (2 seconds), will get 109K~ images; Set 180 (6 seconds), will get 36K~ images.
    We set basic skip_frames as 60 for videos with duration <=5 mins, and skip_frames as 60*3 for duration >5 mins
    '''
    skip_frames = 60

    print("process_and_extract_frames_face3d_cv2read ...")
    # process_and_extract_frames_face3d_cv2read(raw_data_path, sampled_result_path, seq_names, skip_frames, debug=True)
    process_and_extract_frames_face3d_cv2read(raw_data_path, sampled_result_path, seq_names, skip_frames, debug=False)
    
    
