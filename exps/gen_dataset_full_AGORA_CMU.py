
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
    parser.add_argument("--db", type=str, default='./AGORA',
                        help="path to database")
    parser.add_argument("--data_type", type=str, default='train',
                        help="data type, train or val")
    parser.add_argument("--img_size", type=int, default=256,
                        help="output image size")
    parser.add_argument("--plot", type=bool, default=False,
                        help="plot image flag")

    parser.add_argument('--root_dir = ', 
        dest='root_dir', 
        help='root directory of the datasets files', 
        default='./datasets/AGORA/', 
        type=str)
    parser.add_argument('--filename', 
        dest='filename', 
        help='Output filename.',
        default='files_train.txt', 
        type=str)

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
    data_type = args.data_type
    img_size = args.img_size
    isPlot = args.plot
    
    output_path = args.root_dir
    filename = args.filename
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    if "train" == data_type:
        if "AGORA" in mypath:
            img_path = os.path.join(mypath, "images", "train")
            json_path = os.path.join(mypath, "annotations", "coco_style_train_v2.json")
        if "CMU" in mypath:
            img_path = os.path.join(mypath, "images", "train")
            json_path = os.path.join(mypath, "annotations", "coco_style_sampled_train_v2.json")
    if "val" == data_type:
        if "AGORA" in mypath:
            img_path = os.path.join(mypath, "images", "validation")
            json_path = os.path.join(mypath, "annotations", "coco_style_validation_v2.json")
        if "CMU" in mypath:
            img_path = os.path.join(mypath, "images", "val")
            json_path = os.path.join(mypath, "annotations", "coco_style_sampled_val_v2.json")
            
    save_img_path = os.path.join(output_path, data_type)
    save_filename = os.path.join(output_path, filename)

    if os.path.exists(save_img_path):
        shutil.rmtree(save_img_path)
    os.mkdir(save_img_path)

    anno_json_dict = json.load(open(json_path, "r"))
    imgs_dict_list = anno_json_dict["images"]
    imgs_labels_dict = sort_labels_by_image_id(anno_json_dict["annotations"])
    
    print("Json file: %s\n[images number]: %d\n[head instances number]: %d"%(
        json_path, len(imgs_dict_list), len(anno_json_dict["annotations"]) ))
    
    out_imgs = []
    out_poses = []
    
    outfile = open(save_filename, 'w')
    for i, imgs_dict in enumerate(tqdm(imgs_dict_list)):
        img_name = imgs_dict["file_name"]
        img_id = str(imgs_dict["id"])
        
        img_ori = cv2.imread(os.path.join(img_path, img_name))
        
        img_anno_list = imgs_labels_dict[img_id]
        for img_anno in img_anno_list:
            [x, y, w, h] = img_anno["bbox"]
            [pitch, yaw, roll] = img_anno["euler_angles"]
            instance_id = img_anno["id"]
            
            # if abs(yaw) < 90:  # for FSA-Net, we only focus on the head with frontal face
                # img_crop = img_ori[int(y):int(y+h), int(x):int(x+w)]
                # img_crop = cv2.resize(img_crop, (img_size, img_size))

                # out_imgs.append(img_crop)
                # out_poses.append(np.array([yaw, pitch, roll]))
            # else:
                # continue
            
            
            # for 6DRepNet with full-range design, we focus on all the labeled heads
            img_crop = img_ori[int(y):int(y+h), int(x):int(x+w)]
            img_crop = cv2.resize(img_crop, (img_size, img_size))
            
            save_img_path_abs = os.path.join(save_img_path, str(instance_id)+".jpg")
            cv2.imwrite(save_img_path_abs, img_crop)
            
            outfile.write(str(instance_id)+".jpg" + " %.4f %.4f %.4f\n"%(pitch, yaw, roll))


            if i < 2:
                if "AGORA" in mypath:
                    cv2.imwrite("./tmp/"+str(instance_id)+"_agora.jpg", img_crop)
                if "CMU" in mypath:
                    cv2.imwrite("./tmp/"+str(instance_id)+"_cmu.jpg", img_crop)
                    
            # Checking the cropped image
            if isPlot:
                cv2.imshow('check', img_crop)
                k=cv2.waitKey(300)

    outfile.close()
    
if __name__ == "__main__":
    main()
    
'''
AGORA
Json file: /datasdc/zhouhuayi/dataset/AGORA/HPE/annotations/coco_style_train_v2.json
[images number]: 14408
[head instances number]: 105046

Json file: /datasdc/zhouhuayi/dataset/AGORA/HPE/annotations/coco_style_validation_v2.json
[images number]: 1070
[head instances number]: 7505


CMU
Json file: /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/annotations/coco_style_sampled_train_v2.json
[images number]: 15718
[head instances number]: 35725

Json file: /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/annotations/coco_style_sampled_val_v2.json
[images number]: 16216
[head instances number]: 32738
'''