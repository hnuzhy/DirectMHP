import onnx
import torch
import argparse
from utils.torch_utils import select_device
from model import DirectMHPInfer


def save_to_onnx(weights, device, conf_threshold=0.7, iou_threshold=0.45, channels=3, image_size=1280,
                 file_name="DirectMHP.onnx", opset=17):
    device = select_device(device=device, batch_size=1)

    model = DirectMHPInfer(weights=weights, device=device, conf_threshold=conf_threshold,
                           iou_threshold=iou_threshold)

    x = torch.randn(1, channels, image_size, image_size)

    torch.onnx.export(model, x, f"{file_name}.onnx", export_params=True, opset_version=opset,
                      do_constant_folding=True, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    onnx_model = onnx.load(f"{file_name}.onnx")
    onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', default='./weights/agora_m_best.pt', help='path to weights file')
    parser.add_argument('-d', '--device', default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument('-i', '--image-size', type=int, default=1280, help='size of input images')
    parser.add_argument('-c', '--channels', type=int, default=3, help='number of input channels')
    parser.add_argument('-o', '--opset', type=int, default=17, help='opset version')
    parser.add_argument('-n', '--file_name', type=str, default='DirectMHP', help='name of the model without file extension')
    parser.add_argument('--conf-threshold', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45, help='NMS IoU threshold')

    args = parser.parse_args()

    save_to_onnx(**args.__dict__)
