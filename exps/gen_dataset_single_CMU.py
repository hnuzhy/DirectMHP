
import os
import cv2
import json
import argparse
import numpy as np

from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=str, default='./CMU',
                        help="path to database")
    parser.add_argument("--output", type=str, default='./CMU.npz',
                        help="path to output database mat file")
    parser.add_argument("--img_size", type=int, default=64,
                        help="output image size")
    parser.add_argument("--plot", type=bool, default=False,
                        help="plot image flag")


    args = parser.parse_args()
    return args

def sort_labels_by_image_id(labels_list):
    images_labels_dict = {}
    for i, labels_dict in enumerate(labels_list):
        image_id = str(labels_dict['image_id'])
        if image_id in images_labels_dict.keys():
            images_labels_dict[image_id].append(labels_dict)
        else:
            images_labels_dict[image_id] = [labels_dict]
    return images_labels_dict

def main():

    args = get_args()
    mypath = args.db
    output_path = args.output
    img_size = args.img_size
    isPlot = args.plot
    
    if "train" in output_path:
        img_path = os.path.join(mypath, "images", "train")
        json_path = os.path.join(mypath, "annotations", "coco_style_sampled_train_v2.json")
    if "val" in output_path:
        img_path = os.path.join(mypath, "images", "val")
        json_path = os.path.join(mypath, "annotations", "coco_style_sampled_val_v2.json")


    anno_json_dict = json.load(open(json_path, "r"))
    imgs_dict_list = anno_json_dict["images"]
    imgs_labels_dict = sort_labels_by_image_id(anno_json_dict["annotations"])
    
    print("Json file: %s\n[images number]: %d\n[head instances number]: %d"%(
        json_path, len(imgs_dict_list), len(anno_json_dict["annotations"]) ))
    
    out_imgs = []
    out_poses = []
    for i, imgs_dict in enumerate(tqdm(imgs_dict_list)):
        img_name = imgs_dict["file_name"]
        img_id = str(imgs_dict["id"])
        
        img_ori = cv2.imread(os.path.join(img_path, img_name))
        
        img_anno_list = imgs_labels_dict[img_id]
        for img_anno in img_anno_list:
            [x, y, w, h] = img_anno["bbox"]
            [pitch, yaw, roll] = img_anno["euler_angles"]
            instance_id = img_anno["id"]
            
            if abs(yaw) < 90:  # for FSA-Net, we only focus on the head with frontal face
                img_crop = img_ori[int(y):int(y+h), int(x):int(x+w)]
                img_crop = cv2.resize(img_crop, (img_size, img_size))

                out_imgs.append(img_crop)
                out_poses.append(np.array([yaw, pitch, roll]))
            else:
                continue
            
            if i < 2:
                cv2.imwrite("./tmp/"+str(instance_id)+"_cmu.jpg", img_crop)
            
            # Checking the cropped image
            if isPlot:
                cv2.imshow('check', img_crop)
                k=cv2.waitKey(300)

    print("[left head instances]: %d"%(len(out_imgs) ))

    np.savez(output_path, image=np.array(out_imgs), pose=np.array(out_poses), img_size=img_size)

if __name__ == "__main__":
    main()
    
'''
Json file: /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/annotations/coco_style_sampled_train_v2.json
[images number]: 15718
[head instances number]: 35725
[left head instances]: 18447

Json file: /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/annotations/coco_style_sampled_val_v2.json
[images number]: 16216
[head instances number]: 32738
[left head instances]: 16497
'''