import os, os.path as osp
import argparse
import numpy as np
import yaml
from tqdm import tqdm

from pycocotools.coco import COCO

def write_yolov5_labels(data):
    assert not osp.isdir(osp.join(data['path'], data['labels'])), \
        'Labels already generated. Remove or choose new name for labels.'

    splits = [osp.splitext(osp.split(data[s])[-1])[0] for s in ['train', 'val', 'test'] if s in data]
    annotations = [osp.join(data['path'], data['{}_annotations'.format(s)]) for s in ['train', 'val', 'test'] if s in data]
    test_split = [0 if s in ['train', 'val'] else 1 for s in ['train', 'val', 'test'] if s in data]
    img_txt_dir = osp.join(data['path'], data['labels'], 'img_txt')
    os.makedirs(img_txt_dir, exist_ok=True)

    for split, annot, is_test in zip(splits, annotations, test_split):
        img_txt_path = osp.join(img_txt_dir, '{}.txt'.format(split))
        labels_path = osp.join(data['path'], '{}/{}'.format(data['labels'], split))
        if not is_test:
            os.makedirs(labels_path, exist_ok=True)
        coco = COCO(annot)
        if not is_test:
            pbar = tqdm(coco.anns.keys(), total=len(coco.anns.keys()))
            pbar.desc = 'Writing {} labels to {}'.format(split, labels_path)
            for id in pbar:
                a = coco.anns[id]

                if a['image_id'] not in coco.imgs:
                    continue

                if 'train' in split and a['iscrowd']:
                    continue

                img_info = coco.imgs[a['image_id']]
                img_h, img_w = img_info['height'], img_info['width']
                # x, y, w, h = a['head_bbox']
                x, y, w, h = a['bbox']
                xc, yc = x + w / 2, y + h / 2
                xc /= img_w
                yc /= img_h
                w /= img_w
                h /= img_h
                
                [pitch, yaw, roll] = a['euler_angles']
                pitch = (pitch / 180 + 0.5)  # (-90,90)/180 + 0.5 --> (0,1)
                yaw = (yaw / 360 + 0.5)  # (-180, 180)/360 + 0.5 --> (0,1)
                roll = (roll / 180 + 0.5)  # (-90,90)/180 + 0.5 --> (0,1)
                
                yolov5_label_txt = '{}.txt'.format(osp.splitext(img_info['file_name'])[0])
                with open(osp.join(labels_path, yolov5_label_txt), 'a') as f:
                    f.write('{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        0, xc, yc, w, h, pitch, yaw, roll))
            pbar.close()

        with open(img_txt_path, 'w') as f:
            for img_info in coco.imgs.values():
                f.write(osp.join(data['path'], 'images', '{}'.format(split), img_info['file_name']) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/coco-kp.yaml')
    args = parser.parse_args()

    assert osp.isfile(args.data), 'Data config file not found at {}'.format(args.data)

    with open(args.data, 'rb') as f:
        data = yaml.safe_load(f)
    write_yolov5_labels(data)