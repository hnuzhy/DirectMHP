
__author__ = 'Huayi Zhou'

'''

usage:
python compare_error_analysis.py --root-imgdir /path/to/root/imgdir \
    --json-file /path/to/prepared/json/file \
    --anno-file /path/to/annotations/json/file --debug

e.g.:
python compare_error_analysis.py --root-imgdir /datasdc/zhouhuayi/dataset/AGORA/HPE/images/validation --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/agora_m_1280_e300_t40_lw010/weights/val_best_c0.001_i0.65_pd_full.json --anno-file /datasdc/zhouhuayi/dataset/AGORA/HPE/annotations/coco_style_validation.json --debug
[results]
[0, 0, 0, 40.211693506277584, 12.901428962307891, 7.694855970955713, 7.921373673500269, 25.1677368012543, 58.802819890591785, 0, 0, 0]
[18.224194872054188, 12.913802906535974, 11.977283582293937, 12.0253876401812, 8.745764977284157, 9.817249677352743, 8.310038828697268, 9.367154681176574, 10.548681186476957, 13.25264455722177, 12.458162795018275, 20.826818217610814]
[0, 0, 0, 57.99946831484523, 23.82972324396877, 6.409005646864805, 6.30743059098584, 23.58322809309711, 57.96345309420019, 0, 0, 0]


python compare_error_analysis.py --root-imgdir /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/images/val --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/cmu_m_1280_e200_t40_lw010/weights/val_best_c0.001_i0.65_pd_full.json --anno-file /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/annotations/coco_style_sampled_val.json --debug
[results]
[0, 0, 0, 19.9619570872795, 7.406676357656241, 5.374244995360471, 7.009364648555113, 17.976533641153264, 42.68642081816139, 0, 0, 0]
[11.073125970715196, 6.420114565747586, 6.21630644154272, 6.547764549202173, 5.879966301351359, 5.887783334035385, 5.796085305129967, 5.445087367880258, 6.384347982056238, 6.509453387029096, 7.107087266337717, 11.777923735775058]
[0, 0, 0, 37.797477210100105, 15.2464503963692, 4.656056703240891, 4.863564424812517, 15.922617583537287, 41.241533111429845, 0, 0, 0]





usage:
python compare_error_analysis_v2.py --root-imgdir /path/to/root/imgdir \
    --json-file /path/to/prepared/json/file \
    --anno-file /path/to/annotations/json/file --debug

python compare_error_analysis_v2.py --root-imgdir /datasdc/zhouhuayi/dataset/AGORA/HPE/images/validation --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/agora_m_1280_e300_t40_lw010/weights/val_best_c0.001_i0.65_pd_full.json --anno-file /datasdc/zhouhuayi/dataset/AGORA/HPE/annotations/coco_style_validation.json --debug
[results]

[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72.99417527199387, 55.609724412447434, 46.0667093856439, 33.711996765964486, 28.450625149688822, 29.55143144111138, 22.98376501138186, 22.210954318213034, 16.12314467980804, 13.209958393700916, 10.686512257756295, 8.430673989146342, 8.744929807703532, 9.422223243546934, 8.902942640447327, 8.760415548658601, 7.313535783773168, 5.586609852375278, 5.226387526079739, 7.087530809465723, 8.262173050007792, 10.928291714768612, 12.383759862884714, 14.219499400015588, 18.41215575392074, 21.11855597129225, 26.262151147774304, 30.213956849063173, 32.91102585624126, 31.917753554260162, 39.11927736321019, 52.09605832626137, 66.12965986929173, 57.176847674175534, 90.36153793878057, 72.4821484854118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[25.157900051867667, 23.884817586402345, 21.166821052889556, 19.56481710878707, 11.8141285758338, 12.53219983727392, 15.71188531081014, 13.624218581807142, 12.053081426420276, 12.516810219513387, 12.107075882487381, 11.445875643314377, 12.649520983468824, 10.130775248060452, 10.726220094390287, 13.612166129660466, 13.128120513809366, 11.968842014798899, 15.69697426055499, 9.632001322915956, 15.483804570359156, 11.645006058652896, 10.685014095462753, 10.356611304953512, 9.146548266926603, 10.854576567089449, 8.461558460564493, 9.628941784715673, 8.642077529657117, 7.946197651442413, 10.649903095913663, 8.250602441213951, 8.590276737820082, 9.653323319897602, 10.48140081652674, 10.859093277133328, 7.370870084123873, 9.349570546055025, 8.19542361149765, 6.88586955079096, 9.590717238167535, 8.892781161642061, 9.86774471786087, 9.896546916151875, 7.471131001446402, 9.068688193747226, 9.37849094689981, 9.208255506080665, 8.4650290262123, 7.899018471642719, 8.261716740752032, 14.485236608245168, 12.600268510996175, 16.05309267699237, 15.000052008465705, 15.020031064460413, 14.07396474521651, 13.48418367530186, 14.175397070064502, 9.197010843904616, 14.051291662813158, 10.1593986858095, 12.13242075549088, 13.021885602927616, 12.335186512119792, 14.179205818090752, 13.470892604260223, 20.504338738251324, 11.982266286538248, 26.148969590071086, 28.20360181686204, 26.409452940886023]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70.71290410924439, 85.04855440395403, 76.43825545717898, 42.91338700213132, 54.35013665613149, 41.20837838775832, 37.905013312398516, 33.632682351643496, 25.057117899980494, 26.155968369813163, 18.314845456695014, 15.933231999129577, 13.616342044401426, 11.07653083832352, 9.11898179844168, 7.936137716948042, 5.222165256737406, 4.089093956145946, 4.049984216307115, 5.860369585994723, 7.725463115647292, 9.389221707940278, 11.260917115338152, 13.883764845383975, 15.609307228345168, 18.254151099085863, 24.518494297091003, 23.09994606700666, 32.68461364067525, 35.26064807649188, 40.27984699602371, 48.12496088471984, 57.771041347836956, 56.502170314601436, 79.6393935506286, 72.74545198104225, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


python compare_error_analysis_v2.py --root-imgdir /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/images/val --json-file /datasdc/zhouhuayi/face_related/DirectMHP/runs/DirectMHP/cmu_m_1280_e200_t40_lw010/weights/val_best_c0.001_i0.65_pd_full.json --anno-file /datasdc/zhouhuayi/dataset/CMUPanopticDataset/HPE/annotations/coco_style_sampled_val.json --debug
[results]

[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35.40801739153869, 31.589483860751972, 24.192540717854076, 18.933481327029323, 16.223701432954282, 12.658331980141545, 10.078398184454915, 9.079482148819993, 7.708010501100354, 6.782598179266272, 6.555136040303868, 5.875714583592382, 6.01675217975445, 5.815943959516515, 5.3981023741282925, 5.103484539613496, 5.035448046060349, 5.034701765767867, 5.490905604605593, 6.130159301884089, 6.678337900458691, 7.981734852153492, 10.171849984233699, 10.618831341119225, 12.899807957684535, 15.262160034171472, 15.958451621668083, 21.052688296573336, 21.52547324771921, 31.349955187747565, 34.382339585148635, 33.4879663754094, 42.66603209531231, 46.1086990223668, 63.61838861017564, 57.63994375778955, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[17.16226720429872, 11.539396882234234, 9.631266461522456, 10.692474181455333, 8.913612747510987, 6.968545459684506, 8.24488955551861, 6.598828442233448, 6.26222554509559, 5.949327813643982, 6.38436895929868, 5.356234163068045, 5.542367909358609, 6.236013590579165, 6.1046240027114225, 6.200306184375935, 7.473416778881472, 6.633884102555367, 6.631535986885921, 6.89285246349988, 5.549401516426499, 6.712544693058048, 7.163362536851374, 6.524596327246992, 5.734904974945562, 6.171147595007614, 7.100432342632222, 5.883840031470141, 5.4415381109202565, 5.636229841146703, 6.104062081281202, 4.995969700054788, 4.8147169336731706, 4.558947898593563, 6.3130373715655645, 9.030331377218438, 7.81504288018686, 4.562655887359898, 5.570725555169311, 6.532252311518441, 4.958754800103939, 5.94835612607347, 4.860651630743885, 5.575086701271192, 5.651245028624852, 5.669233832809818, 5.172316640939801, 5.65052521331266, 5.85785197380578, 5.573997780907589, 6.338178639257099, 7.537182193276217, 6.55094088229122, 6.475750335255333, 5.015371621167982, 6.640488995888127, 6.194886081464597, 6.471868066318264, 7.308436300829931, 6.233524865046266, 5.8229739753034195, 7.048351688473666, 7.060110757548085, 7.515613157763315, 7.447130455104933, 7.804115835773275, 7.785024438777153, 7.901861351465182, 11.746343398316226, 11.615900343115733, 13.443213496315414, 16.901764944513143]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60.75053325359367, 50.70591522223075, 42.469127154068495, 32.39916272483847, 28.310132677196368, 24.976241968171923, 22.974322907418603, 17.701588784900604, 15.27061551800479, 16.759874203390382, 14.127594159861362, 11.356579831849565, 10.08158837466275, 8.522273080155067, 6.280971546290719, 4.93558385562088, 3.9685152605752245, 3.2015491115827452, 3.144199309529352, 4.068804456819185, 5.285678921324436, 7.01659497784365, 9.101716787443786, 10.15976952314743, 12.022647221014966, 14.930348683289608, 16.60482469046746, 16.365226080765037, 19.0848588694588, 22.320243041949546, 30.186302629141544, 32.535210544449576, 33.7474322168678, 44.895065734976434, 61.29272904975674, 59.74996230208941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

'''

import cv2
import os
import time
import json
import argparse
from tqdm import tqdm

import shutil
import numpy as np

from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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

def main(args):

    if os.path.exists("./tmp/"):
        shutil.rmtree("./tmp/")
    os.mkdir("./tmp/")
        
    with open(args.json_file, "r") as json_f:
        pd_results_list = json.load(json_f)
        
    with open(args.anno_file, "r") as json_f:
        gt_results_list = json.load(json_f)
    images_labels_dict = sort_labels_by_image_id(gt_results_list['annotations'])

    # interval = 20  # 10 or 15 is better
    # interval = 30
    # interval = 10
    interval = 5
    bins = 360 // interval
    MAE_list_sum = [[0]*bins, [0]*bins, [0]*bins]  # pitch, yaw, roll
    MAE_list_cnt = [[0]*bins, [0]*bins, [0]*bins]  # pitch, yaw, roll
    images_pd_results_dict = {}

    for ind, pd_results in enumerate(tqdm(pd_results_list)):
        # if ind > 300: break

        temp_pitch = abs(pd_results['gt_pitch'] - pd_results['pitch'])
        temp_yaw = abs(pd_results['gt_yaw'] - pd_results['yaw'])
        temp_yaw = min(temp_yaw, 360 - temp_yaw)
        temp_roll = abs(pd_results['gt_roll'] - pd_results['roll'])
        
        MAE_list_sum[0][int((pd_results['gt_pitch'] + 180)//interval)] += temp_pitch
        MAE_list_cnt[0][int((pd_results['gt_pitch'] + 180)//interval)] += 1
        MAE_list_sum[1][int((pd_results['gt_yaw'] + 180)//interval)] += temp_yaw
        MAE_list_cnt[1][int((pd_results['gt_yaw'] + 180)//interval)] += 1
        MAE_list_sum[2][int((pd_results['gt_roll'] + 180)//interval)] += temp_roll
        MAE_list_cnt[2][int((pd_results['gt_roll'] + 180)//interval)] += 1

        if str(pd_results["image_id"]) in images_pd_results_dict:
            images_pd_results_dict[str(pd_results["image_id"])].append(pd_results)
        else:
            images_pd_results_dict[str(pd_results["image_id"])] = [pd_results]
            
    MAE_list_0 = [i/j if j!=0 else 0 for i,j in zip(MAE_list_sum[0], MAE_list_cnt[0])]
    MAE_list_1 = [i/j if j!=0 else 0 for i,j in zip(MAE_list_sum[1], MAE_list_cnt[1])]
    MAE_list_2 = [i/j if j!=0 else 0 for i,j in zip(MAE_list_sum[2], MAE_list_cnt[2])]
    
    print(MAE_list_0)
    print(MAE_list_1)
    print(MAE_list_2)

    if args.debug:
        # choose those undetected hard head examples
        matched_iou_threshold = 0.5
        crop_padding = 1.0
        total_count = 0
        for image_id, labels in tqdm(images_labels_dict.items()):
            if image_id not in images_pd_results_dict:
                # this image have none of one head having been detected
                img_path = os.path.join(args.root_imgdir, image_id+".jpg")
                img_ori = cv2.imread(img_path)
                for label in labels:
                    bbox = label['bbox']  # bbox default format is [x0,y0,w,h], should be converted to [x0,y0,x1,y1]
                    bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                    cv2.rectangle(img_ori, (int(bbox[0]), int(bbox[1])), 
                        (int(bbox[2]), int(bbox[3])), (255,255,255), 2)
                    
                    save_img_path = "./tmp/"+str(label['id'])+".jpg"
                    crop_x0 = max(int(bbox[0] - (bbox[2] - bbox[0]) * crop_padding), 0)
                    crop_y0 = max(int(bbox[1] - (bbox[3] - bbox[1]) * crop_padding), 0)
                    crop_x1 = min(int(bbox[2] + (bbox[2] - bbox[0]) * crop_padding), img_ori.shape[1]-1)
                    crop_y1 = min(int(bbox[3] + (bbox[3] - bbox[1]) * crop_padding), img_ori.shape[0]-1)
                    # get a square face subimage
                    cen_x, cen_y = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
                    halflen = min(min(cen_x-crop_x0, crop_x1-cen_x), min(cen_y-crop_y0, crop_y1-cen_y))
                    crop_x0, crop_x1 = cen_x - halflen, cen_x + halflen
                    crop_y0, crop_y1 = cen_y - halflen, cen_y + halflen
                    cv2.imwrite(save_img_path, img_ori[crop_y0:crop_y1, crop_x0:crop_x1])
                    total_count += 1
            else:
                # check if it has at least one head not been detected
                undetected_head_list = []
                for label in labels:
                    gt_bbox = label['bbox']
                    max_iou, matched_index = 0, -1
                    for i, pd_result in enumerate(images_pd_results_dict[image_id]):
                        pd_bbox = pd_result['bbox']
                        temp_iou = calculate_bbox_iou(pd_bbox, gt_bbox, format='xywh')
                        if temp_iou > max_iou:
                            max_iou = temp_iou
                            matched_index = i
                    if max_iou < matched_iou_threshold:
                        undetected_head_list.append(gt_bbox + [label['id']])
                        
                if len(undetected_head_list) != 0:
                    img_path = os.path.join(args.root_imgdir, image_id+".jpg")
                    img_ori = cv2.imread(img_path)
                    for bbox in undetected_head_list:
                        id = bbox[-1]
                        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                        cv2.rectangle(img_ori, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), (255,255,255), 2)
                        save_img_path = "./tmp/"+str(id)+".jpg"
                        crop_x0 = max(int(bbox[0] - (bbox[2] - bbox[0]) * crop_padding), 0)
                        crop_y0 = max(int(bbox[1] - (bbox[3] - bbox[1]) * crop_padding), 0)
                        crop_x1 = min(int(bbox[2] + (bbox[2] - bbox[0]) * crop_padding), img_ori.shape[1]-1)
                        crop_y1 = min(int(bbox[3] + (bbox[3] - bbox[1]) * crop_padding), img_ori.shape[0]-1)
                        # get a square face subimage
                        cen_x, cen_y = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
                        halflen = min(min(cen_x-crop_x0, crop_x1-cen_x), min(cen_y-crop_y0, crop_y1-cen_y))
                        crop_x0, crop_x1 = cen_x - halflen, cen_x + halflen
                        crop_y0, crop_y1 = cen_y - halflen, cen_y + halflen
                        cv2.imwrite(save_img_path, img_ori[crop_y0:crop_y1, crop_x0:crop_x1])
                        total_count += 1
            
            if total_count > 1000:  # we do not want to save to many undetected heads examples
                break
                        
    else:
        '''AGORA Euler Angels Stat'''
        # plt.figure(figsize=(15, 5), dpi=100)
        plt.figure(figsize=(15, 4), dpi=100)
        if 'AGORA' in args.root_imgdir:
            plt.title("AGORA-HPE MAE Errors")
        if 'CMU' in args.root_imgdir:
            plt.title("CMU-HPE MAE Errors")
        plt.bar(-180+(np.arange(bins)+1/5+1/10)*interval, MAE_list_0, width=interval/5, color='r', label="Pitch")
        plt.bar(-180+(np.arange(bins)+2/5+1/10)*interval, MAE_list_1, width=interval/5, color='g', label="Yaw")
        plt.bar(-180+(np.arange(bins)+3/5+1/10)*interval, MAE_list_2, width=interval/5, color='b', label="Roll")
        
        avg_x = np.array(-180 + (np.arange(bins)+2/5+1/10) * interval)
        avg_y = (np.array(MAE_list_0) + np.array(MAE_list_1) + np.array(MAE_list_2)) / 3
        # cubic_interpolation_model = interp1d(avg_x, avg_y, kind = "cubic")
        X_Y_Spline = make_interp_spline(avg_x, avg_y, k=1)  # larger k will make curve more "wiggly"
        X_ = np.linspace(avg_x.min(), avg_x.max(), 500)  # Plotting the Graph
        # Y_ = cubic_interpolation_model(X_)
        Y_ = X_Y_Spline(X_)
        plt.plot(X_, Y_, linewidth=8, alpha=0.4, color='c', label="Avg")

        plt.legend(prop ={'size': 20})
        # plt.xticks(range(-180,181,interval), rotation=0)
        plt.xticks(range(-180,181,interval*3), rotation=0)
        plt.ylabel('MAE')
        plt.xlabel('Degree')
        if 'AGORA' in args.root_imgdir:
            save_name = 'AGORA-HPE_MAE_Errors_v2.png'
        if 'CMU' in args.root_imgdir:
            save_name = 'CMU-HPE_MAE_Errors_v2.png'

        plt.savefig(save_name)
        img = cv2.imread(save_name)
        img_cut = img[42:380, 136:1355]
        cv2.imwrite(save_name, img_cut)
        

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    
    parser.add_argument('--root-imgdir', default='',
                        help='root path to multiple images')
    parser.add_argument('--json-file', default='',
                        help='json file path that contains multiple images and their head bboxes')
    parser.add_argument('--anno-file', default='',
                        help='json file path that contains ground-truths of validation set')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--debug',  action='store_true', help='whether set into debug mode')
    
    args = parser.parse_args()
    main(args)