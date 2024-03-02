import onnx
import torch
import argparse
from utils.torch_utils import select_device
from model import DirectMHPInfer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', default='./weights/agora_m_best.pt', help='path to weights file')
    parser.add_argument('-d', '--device', default="cuda:0" if torch.cuda.is_available() else "cpu", help='cuda device, i.e. 0 or cpu')
    parser.add_argument('-i', '--imagesize', type=int, default=1280, help='size of input images')
    parser.add_argument('-c', '--channels', type=int, default=3, help='number of input channels')
    parser.add_argument('-o', '--opset', type=int, default=17, help='opset version')
    parser.add_argument('-n', '--name', type=str, default='DirectMHP', help='name of the model without file extension')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')

    args = parser.parse_args()

    device = select_device(device=args.device, batch_size=1)
    print('Using device: {}'.format(device))
    print('Weights path: {}'.format(args.weights))

    model = DirectMHPInfer(weights=args.weights, device=args.device, conf_threshold=args.conf_thres,
                           iou_threshold=args.iou_thres)

    # model = attempt_load(args.weights, map_location=device)
    x = torch.randn(1, args.channels, args.imagesize, args.imagesize)

    torch.onnx.export(model, x, f"{args.name}.onnx", export_params=True, opset_version=args.opset,
                      do_constant_folding=True, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    onnx_model = onnx.load(f"{args.name}.onnx")
    onnx.checker.check_model(onnx_model)
