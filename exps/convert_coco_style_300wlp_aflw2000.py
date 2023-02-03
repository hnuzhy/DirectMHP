import os
import json
import copy
import shutil
from tqdm import tqdm

coco_dict_template = {
    'info': {
        'description': 'Face landmarks, Euler angles and 3D Cubes of 300W_LP & AFLW2000 Dataset',
        'url': 'http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm',
        'version': '1.0',
        'year': 2022,
        'contributor': 'Huayi Zhou',
        'date_created': '2022/07/28',
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

def convert_to_coco_style(source_img, target_img, source_json, target_json, coco_dict):
    print(source_img, " --> ", target_img)
    print(source_json, " --> ", target_json)
    
    if os.path.exists(target_img):
        shutil.rmtree(target_img)
    os.mkdir(target_img)
    
    json_img_dict = json.load(open(source_json, "r"))
    index_id = 0
    for img_name in tqdm(json_img_dict.keys()):
        labels = json_img_dict[img_name]
        
        image_id = 1000000 + index_id  # 300W_LP has about 122217 images
        temp_image = {'file_name': str(image_id)+".jpg", 
            'height': labels['height'], 'width': labels['width'], 'id': image_id}
        
        source_img_path = os.path.join(source_img, img_name)
        target_img_path = os.path.join(target_img, str(image_id)+".jpg")
        shutil.copy(source_img_path, target_img_path)
        
        # bbox: [xmin, ymin, xmax, ymax] --> [xmin, ymin, w, h]
        [xmin, ymin, xmax, ymax] = labels["bbox"]
        labels["bbox"] = [xmin, ymin, xmax-xmin, ymax-ymin]
        
        # pose: [yaw, pitch, roll] --> [pitch, yaw, roll]
        [yaw, pitch, roll] = labels["pose"]
        labels["pose"] = [pitch, yaw, roll]
        
        labels_new = {
            'face2d_pts': labels["landmarks"],
            'bbox': labels["bbox"],
            'euler_angles': labels["pose"], 
            'cube': labels["cube"],
            'image_id': image_id,
            'id': image_id,  # only one head in each image
            'category_id': 1,
            'iscrowd': 0,
            'segmentation': [],  # This script is not for segmentation
            'area': round(labels["bbox"][-1] * labels["bbox"][-2], 4)
        }
        coco_dict['images'].append(temp_image)
        coco_dict['annotations'].append(labels_new)
        
        index_id += 1
        
    with open(target_json, "w") as dst_ann_file:
        json.dump(coco_dict, dst_ann_file)
        
    
if __name__ == '__main__':

    train_image_file = "./HeadCube3D/images/300W_LP/"
    train_image_file_coco = "./HeadCube3D/images/train/"
    train_json_file = "./HeadCube3D/annotations/train_300W_LP.json"
    train_json_file_coco = "./HeadCube3D/annotations/train_300W_LP_coco_style.json"
    coco_dict_train = copy.deepcopy(coco_dict_template)
    convert_to_coco_style(train_image_file, train_image_file_coco,
        train_json_file, train_json_file_coco, coco_dict_train)
    
    
    val_image_file = "./HeadCube3D/images/AFLW2000/"
    val_image_file_coco = "./HeadCube3D/images/validation/"
    val_json_file = "./HeadCube3D/annotations/val_AFLW2000.json"
    val_json_file_coco = "./HeadCube3D/annotations/val_AFLW2000_coco_style.json"
    coco_dict_val = copy.deepcopy(coco_dict_template)
    convert_to_coco_style(val_image_file, val_image_file_coco,
        val_json_file, val_json_file_coco, coco_dict_val)
    