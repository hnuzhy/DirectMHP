import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import argparse
import torch
import cv2
import yaml
import imageio
from tqdm import tqdm
import os.path as osp
import numpy as np

from utils.torch_utils import select_device, time_sync
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from utils.plots import plot_3axis_Zaxis
from models.experimental import attempt_load


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # video options
    parser.add_argument('-p', '--video-path', default='', help='path to video file')

    parser.add_argument('--data', type=str, default='data/agora_coco.yaml')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--save-size', type=int, default=1080)
    parser.add_argument('--weights', default='yolov5m6.pt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    
    parser.add_argument('--start', type=int, default=0, help='start time (s)')
    parser.add_argument('--end', type=int, default=-1, help='end time (s), -1 for remainder of video')
    parser.add_argument('--color', type=int, nargs='+', default=[255, 255, 255], help='head bbox color')
    parser.add_argument('--thickness', type=int, default=2, help='thickness of Euler angle lines')
    parser.add_argument('--alpha', type=float, default=0.4, help='head bbox and head pose alpha')
    
    parser.add_argument('--display', action='store_true', help='display inference results')
    parser.add_argument('--fps-size', type=int, default=1)
    parser.add_argument('--gif', action='store_true', help='create gif')
    parser.add_argument('--gif-size', type=int, nargs='+', default=[480, 270])

    args = parser.parse_args()

    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    device = select_device(args.device, batch_size=1)
    print('Using device: {}'.format(device))

    model = attempt_load(args.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    dataset = LoadImages(args.video_path, img_size=imgsz, stride=stride, auto=True)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


    cap = dataset.cap
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.end == -1:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - fps * args.start)
    else:
        n = int(fps * (args.end - args.start))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gif_frames = []
    out_path = '{}_{}'.format(osp.splitext(args.video_path)[0], "DirectMHP")
    print("fps:", fps, "\t total frames:", n, "\t out_path:", out_path)

    write_video = not args.display and not args.gif
    if write_video:
        # writer = cv2.VideoWriter(out_path + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        writer = cv2.VideoWriter(out_path + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, 
            (int(args.save_size*w/h), args.save_size))

    dataset = tqdm(dataset, desc='Running inference', total=n)
    t0 = time_sync()
    for i, (path, img, im0, _) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        out_ori = model(img, augment=True, scales=args.scales)[0]
        out = non_max_suppression(out_ori, args.conf_thres, args.iou_thres, num_angles=data['num_angles'])
        # predictions (Array[N, 9]), x1, y1, x2, y2, conf, class, pitch, yaw, roll
        bboxes = scale_coords(img.shape[2:], out[0][:, :4], im0.shape[:2]).cpu().numpy()  # native-space pred
        scores = out[0][:, 4].cpu().numpy() 
        pitchs_yaws_rolls = out[0][:, 6:].cpu().numpy()   # N*3
        
        im0_copy = im0.copy()
        
        # draw head bboxes and pose
        for j, [x1, y1, x2, y2] in enumerate(bboxes):
            im0_copy = cv2.rectangle(im0_copy, (int(x1), int(y1)), (int(x2), int(y2)), 
                args.color, thickness=args.thickness)
            # im0_copy = cv2.putText(im0_copy, str(round(scores[j], 3)), (int(x1), int(y1)), 
                # cv2.FONT_HERSHEY_PLAIN, 0.7, (255,255,255), thickness=2)
            pitch = (pitchs_yaws_rolls[j][0] - 0.5) * 180
            yaw = (pitchs_yaws_rolls[j][1] - 0.5) * 360
            roll = (pitchs_yaws_rolls[j][2] - 0.5) * 180
            im0_copy = plot_3axis_Zaxis(im0_copy, yaw, pitch, roll, tdx=(x1+x2)/2, tdy=(y1+y2)/2, 
                size=max(y2-y1, x2-x1)*0.8, thickness=args.thickness)
                
        im0 = cv2.addWeighted(im0, args.alpha, im0_copy, 1 - args.alpha, gamma=0)

        if i == 0:
            t = time_sync() - t0
        else:
            t = time_sync() - t1

        if not args.gif and args.fps_size:
            cv2.putText(im0, '{:.1f} FPS'.format(1 / t), (5 * args.fps_size, 25 * args.fps_size),
                cv2.FONT_HERSHEY_SIMPLEX, args.fps_size, (255, 255, 255), thickness=2 * args.fps_size)

        if args.gif:
            gif_img = cv2.cvtColor(cv2.resize(im0, dsize=tuple(args.gif_size)), cv2.COLOR_RGB2BGR)
            if args.fps_size:
                cv2.putText(gif_img, '{:.1f} FPS'.format(1 / t), (5 * args.fps_size, 25 * args.fps_size),
                    cv2.FONT_HERSHEY_SIMPLEX, args.fps_size, (255, 255, 255), thickness=2 * args.fps_size)
            gif_frames.append(gif_img)
        elif write_video:
            im0 = cv2.resize(im0, dsize=(int(args.save_size*w/h), args.save_size))
            writer.write(im0)
        else:
            cv2.imshow('', im0)
            cv2.waitKey(1)

        t1 = time_sync()
        if i == n - 1:
            break

    cv2.destroyAllWindows()
    cap.release()
    if write_video:
        writer.release()

    if args.gif:
        print('Saving GIF...')
        with imageio.get_writer(out_path + '.gif', mode="I", fps=fps) as writer:
            for idx, frame in tqdm(enumerate(gif_frames)):
                writer.append_data(frame)

