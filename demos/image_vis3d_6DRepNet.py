import sys
sys.path.append('../')

from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import argparse
import yaml
import cv2
import math
from math import cos, sin
import os.path as osp
import numpy as np
from PIL import Image

from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load

from scipy.spatial.transform import Rotation
from utils.renderer import Renderer


'''from https://github.com/thohemp/6DRepNet'''

#input batch*4*4 or batch*3*3
#output torch batch*3 x, y, z in radiant
#the rotation is in the sequence of x,y,z
# https://learnopencv.com/rotation-matrix-to-euler-angles/
def compute_euler_angles_from_rotation_matrices(rotation_matrices, full_range=False, use_gpu=True, gpu_id=0):
    batch=rotation_matrices.shape[0]
    R=rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular= sy<1e-6
    singular=singular.float()
    
    
    '''2023.01.15'''
    for i in range(len(sy)):  # expand y (yaw angle) range into (-180, 180)
        if R[i,0,0] < 0 and full_range:
            sy[i] = -sy[i]
    
    
    x=torch.atan2(R[:,2,1], R[:,2,2])
    y=torch.atan2(-R[:,2,0], sy)  # sy > 0, so y (yaw angle) is always in range (-90, 90)
    z=torch.atan2(R[:,1,0],R[:,0,0])
    
    xs=torch.atan2(-R[:,1,2], R[:,1,1])
    ys=torch.atan2(-R[:,2,0], sy)  # sy > 0, so y (yaw angle) is always in range (-90, 90)
    zs=R[:,1,0]*0
        
    if use_gpu:
        out_euler=torch.autograd.Variable(torch.zeros(batch,3).cuda(gpu_id))
    else:
        out_euler=torch.autograd.Variable(torch.zeros(batch,3))  
    out_euler[:,0]=x*(1-singular)+xs*singular
    out_euler[:,1]=y*(1-singular)+ys*singular
    out_euler[:,2]=z*(1-singular)+zs*singular
        
    return out_euler

# batch*n
def normalize_vector( v, use_gpu=True, gpu_id = 0):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    if use_gpu:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda(gpu_id)))
    else:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))  
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
    
# poses batch*6
def compute_rotation_matrix_from_ortho6d(poses, use_gpu=True, gpu_id=0):
    x_raw = poses[:,0:3]#batch*3
    y_raw = poses[:,3:6]#batch*3

    x = normalize_vector(x_raw, use_gpu, gpu_id=gpu_id) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z, use_gpu,gpu_id=gpu_id)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix


# https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle


    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
        #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)


    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

def create_RepVGG_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_A1(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_A2(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)

def create_RepVGG_B0(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B1g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B2g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B2g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B3(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B3g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B3g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)

def create_RepVGG_D2se(deploy=False):
    return RepVGG(num_blocks=[8, 14, 24, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_se=True)


func_dict = {
'RepVGG-A0': create_RepVGG_A0,
'RepVGG-A1': create_RepVGG_A1,
'RepVGG-A2': create_RepVGG_A2,
'RepVGG-B0': create_RepVGG_B0,
'RepVGG-B1': create_RepVGG_B1,
'RepVGG-B1g2': create_RepVGG_B1g2,
'RepVGG-B1g4': create_RepVGG_B1g4,
'RepVGG-B2': create_RepVGG_B2,
'RepVGG-B2g2': create_RepVGG_B2g2,
'RepVGG-B2g4': create_RepVGG_B2g4,
'RepVGG-B3': create_RepVGG_B3,
'RepVGG-B3g2': create_RepVGG_B3g2,
'RepVGG-B3g4': create_RepVGG_B3g4,
'RepVGG-D2se': create_RepVGG_D2se,      #   Updated at April 25, 2021. This is not reported in the CVPR paper.
}
def get_RepVGG_func_by_name(name):
    return func_dict[name]


class SixDRepNet(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 bins=(1, 2, 3, 6),
                 droBatchNorm=nn.BatchNorm2d,
                 pretrained=True, 
                 gpu_id=0):
        super(SixDRepNet, self).__init__()
        self.gpu_id = gpu_id
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
                    v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel

        self.linear_reg = nn.Linear(fea_dim, 6)

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x= self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        if self.gpu_id ==-1:
            return compute_rotation_matrix_from_ortho6d(x, False, self.gpu_id)
        else:
            return compute_rotation_matrix_from_ortho6d(x, True, self.gpu_id)



'''from https://github.com/vitoralbiero/img2pose'''

def bbox_is_dict(bbox):
    # check if the bbox is a not dict and convert it if needed
    if not isinstance(bbox, dict):
        temp_bbox = {}
        temp_bbox["left"] = bbox[0]
        temp_bbox["top"] = bbox[1]
        temp_bbox["right"] = bbox[2]
        temp_bbox["bottom"] = bbox[3]
        bbox = temp_bbox

    return bbox

def get_bbox_intrinsics(image_intrinsics, bbox):
    # crop principle point of view
    bbox_center_x = bbox["left"] + ((bbox["right"] - bbox["left"]) // 2)
    bbox_center_y = bbox["top"] + ((bbox["bottom"] - bbox["top"]) // 2)

    # create a camera intrinsics from the bbox center
    bbox_intrinsics = image_intrinsics.copy()
    bbox_intrinsics[0, 2] = bbox_center_x
    bbox_intrinsics[1, 2] = bbox_center_y

    return bbox_intrinsics

def pose_bbox_to_full_image(pose, image_intrinsics, bbox):
    # check if bbox is np or dict
    bbox = bbox_is_dict(bbox)

    # rotation vector
    rvec = pose[:3].copy()

    # translation and scale vector
    tvec = pose[3:].copy()

    # get camera intrinsics using bbox
    bbox_intrinsics = get_bbox_intrinsics(image_intrinsics, bbox)

    # focal length
    focal_length = image_intrinsics[0, 0]

    # bbox_size
    bbox_width = bbox["right"] - bbox["left"]
    bbox_height = bbox["bottom"] - bbox["top"]
    bbox_size = bbox_width + bbox_height
    bbox_size *= 0.5 * 0.5

    # adjust scale
    tvec[2] *= focal_length / bbox_size

    # project crop points using the crop camera intrinsics
    projected_point = bbox_intrinsics.dot(tvec.T)

    # reverse the projected points using the full image camera intrinsics
    tvec = projected_point.dot(np.linalg.inv(image_intrinsics.T))

    # same for rotation
    rmat = Rotation.from_rotvec(rvec).as_matrix()
    # project crop points using the crop camera intrinsics
    projected_point = bbox_intrinsics.dot(rmat)
    # reverse the projected points using the full image camera intrinsics
    rmat = np.linalg.inv(image_intrinsics).dot(projected_point)
    rvec = Rotation.from_matrix(rmat).as_rotvec()

    return np.concatenate([rvec, tvec])

def convert_euler_bbox_to_6dof(euler_angle, bbox, global_intrinsics):
    
    [pitch, yaw, roll] = euler_angle
    ideal_angle = [pitch, -yaw, -roll]
    rot_mat = Rotation.from_euler('xyz', ideal_angle, degrees=True).as_matrix()
    rot_mat_2 = np.transpose(rot_mat)
    rotvec = Rotation.from_matrix(rot_mat_2).as_rotvec()
    
    local_pose = np.array([rotvec[0], rotvec[1], rotvec[2], 0, 0, 1])
    
    global_pose_6dof = pose_bbox_to_full_image(local_pose, global_intrinsics, bbox_is_dict(bbox))

    return global_pose_6dof.tolist()


def plot_3axis_Zaxis(img, yaw, pitch, roll, tdx=None, tdy=None, size=50., limited=True, thickness=2, extending=False):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    
    if tdx != None and tdy != None:
        face_x = tdx
        face_y = tdy
    else:
        height, width = img.shape[:2]
        face_x = width / 2
        face_y = height / 2

    # X-Axis (pointing to right) drawn in red
    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    
    # Y-Axis (pointing to down) drawn in green
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    if extending:
        # Plot head oritation extended line in yellow
        # scale_ratio = 5
        scale_ratio = 2
        base_len = math.sqrt((face_x - x3)**2 + (face_y - y3)**2)
        if face_x == x3:
            endx = tdx
            if face_y < y3:
                if limited:
                    endy = tdy + (y3 - face_y) * scale_ratio
                else:
                    endy = img.shape[0]
            else:
                if limited:
                    endy = tdy - (face_y - y3) * scale_ratio
                else:
                    endy = 0
        elif face_x > x3:
            if limited:
                endx = tdx - (face_x - x3) * scale_ratio
                endy = tdy - (face_y - y3) * scale_ratio
            else:
                endx = 0
                endy = tdy - (face_y - y3) / (face_x - x3) * tdx
        else:
            if limited:
                endx = tdx + (x3 - face_x) * scale_ratio
                endy = tdy + (y3 - face_y) * scale_ratio
            else:
                endx = img.shape[1]
                endy = tdy - (face_y - y3) / (face_x - x3) * (tdx - endx)
        # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (0,0,0), 2)
        # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (255,255,0), 2)
        cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (0,255,255), thickness)

    # X-Axis pointing to right. drawn in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),thickness)
    # Y-Axis pointing to down. drawn in green    
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,255,0),thickness)
    # Z-Axis (out of the screen) drawn in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),thickness)

    return img


def crop_image(img, bbox, scale=1.0):
# def crop_image(img_path, bbox, scale=1.0):
    # img = cv2.imread(img_path)
    img_h, img_w, img_c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x_min, y_min, x_max, y_max = bbox
    # img_rgb = img_rgb[y_min:y_max, x_min:x_max]
    
    # enlarge the head/face bounding box
    # scale_ratio = 1.25
    scale_ratio = scale
    center_x, center_y = (x_max + x_min)/2, (y_max + y_min)/2
    face_w, face_h = x_max - x_min, y_max - y_min
    # new_w, new_h = face_w*scale_ratio, face_h*scale_ratio
    new_w = max(face_w*scale_ratio, face_h*scale_ratio)
    new_h = max(face_w*scale_ratio, face_h*scale_ratio)
    new_x_min = max(0, int(center_x-new_w/2))
    new_y_min = max(0, int(center_y-new_h/2))
    new_x_max = min(img_w-1, int(center_x+new_w/2))
    new_y_max = min(img_h-1, int(center_y+new_h/2))
    img_rgb = img_rgb[new_y_min:new_y_max, new_x_min:new_x_max]
    
    left = max(0, -int(center_x-new_w/2))
    top = max(0, -int(center_y-new_h/2))
    right = max(0, int(center_x+new_w/2) - img_w + 1)
    bottom = max(0, int(center_y+new_h/2) - img_h + 1)
    img_rgb = cv2.copyMakeBorder(img_rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    return img_rgb
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--img-path', default='test_imgs/100024.jpg', help='path to image or dir')
    parser.add_argument('--data', type=str, default='data/agora_coco.yaml')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--weights', default='yolov5m6.pt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--thickness', type=int, default=2, help='thickness of Euler angle lines')

    args = parser.parse_args()
    
    ''' Create the renderer for 3D face/head visualization '''   
    renderer = Renderer(
        vertices_path="pose_references/vertices_trans.npy", 
        triangles_path="pose_references/triangles.npy"
    )

    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    device = select_device(args.device, batch_size=1)
    print('Using device: {}'.format(device))

    model_DirectMHP = attempt_load(args.weights, map_location=device)
    stride = int(model_DirectMHP.stride.max())  # model stride
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    dataset = LoadImages(args.img_path, img_size=imgsz, stride=stride, auto=True)
    dataset_iter = iter(dataset)
    
    
    gpu_id = 0
    img_H, img_W = 256, 256
    # snapshot_path = "../6DRepNet/sixdrepnet/weights/6DRepNet_300W_LP_AFLW2000.pth"
    snapshot_path = "../6DRepNet/sixdrepnet/output/SixDRepNet_AGORA_bs256_e100/epoch_last.pth"
    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])])
    model_6dRepNet = SixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='',
                        deploy=True,
                        pretrained=False,
                        gpu_id=gpu_id)          
    saved_state_dict = torch.load(snapshot_path, map_location='cpu')
    if 'model_state_dict' in saved_state_dict:
        model_6dRepNet.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model_6dRepNet.load_state_dict(saved_state_dict)    
    model_6dRepNet.cuda(gpu_id)
    model_6dRepNet.eval()
    
    
    for index in range(len(dataset)):
        
        (single_path, img, im0, _) = next(dataset_iter)
        
        if '_res' in single_path: continue
        
        print(index, single_path, "\n")
        
        img = torch.from_numpy(img).to(device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        out_ori = model_DirectMHP(img, augment=True, scales=args.scales)[0]
        out = non_max_suppression(out_ori, args.conf_thres, args.iou_thres, num_angles=data['num_angles'])

        (h, w, c) = im0.shape
        global_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
        global_poses, global_poses_6dRepNet = [], []
        
        # predictions (Array[N, 9]), x1, y1, x2, y2, conf, class, pitch, yaw, roll
        bboxes = scale_coords(img.shape[2:], out[0][:, :4], im0.shape[:2]).cpu().numpy()  # native-space pred
        scores = out[0][:, 4].cpu().numpy() 
        pitchs_yaws_rolls = out[0][:, 6:].cpu().numpy()   # N*3
        euler_angles, euler_angles_6dRepNet = [], []
        for i, [x1, y1, x2, y2] in enumerate(bboxes):
            # head pose results by our method DirectMHP
            pitch = (pitchs_yaws_rolls[i][0] - 0.5) * 180
            yaw = (pitchs_yaws_rolls[i][1] - 0.5) * 360
            roll = (pitchs_yaws_rolls[i][2] - 0.5) * 180

            euler_angle = [pitch, yaw, roll]
            bbox = [x1, y1, x2, y2]
            global_pose = convert_euler_bbox_to_6dof(euler_angle, bbox, global_intrinsics)
            global_poses.append(global_pose)
            euler_angles.append(euler_angle)
            
            
            '''Running 6dRepNet'''
            croped_frame = crop_image(im0, bbox, scale=1.0)
            croped_resized_frame = cv2.resize(croped_frame, (img_W, img_H))  # h,w -> 256,256
            
            img_rgb = croped_resized_frame[..., ::-1]  # bgr --> rgb
            PIL_image = Image.fromarray(img_rgb)  # numpy array --> PIL image
            img_input = transformations(PIL_image)
            img_input = torch.Tensor(img_input).cuda(gpu_id)
            
            R_pred = model_6dRepNet(img_input.unsqueeze(0))  # hwc --> nhwc
            euler = compute_euler_angles_from_rotation_matrices(R_pred, full_range=True)*180/np.pi
            p_pred_deg = euler[:, 0].cpu().detach().numpy()
            y_pred_deg = euler[:, 1].cpu().detach().numpy()
            r_pred_deg = euler[:, 2].cpu().detach().numpy()
            
            yaw, pitch, roll = y_pred_deg[0], p_pred_deg[0], r_pred_deg[0]
            
            if yaw > 360: yaw = yaw - 360
            if yaw < -360: yaw = yaw + 360
            if pitch > 360: pitch = pitch - 360
            if pitch < -360: pitch = pitch + 360
            if roll > 360: roll = roll - 360
            if roll < -360: roll = roll + 360

            euler_angle = [pitch, yaw, roll]
            global_pose = convert_euler_bbox_to_6dof(euler_angle, bbox, global_intrinsics)
            global_poses_6dRepNet.append(global_pose)
            euler_angles_6dRepNet.append(euler_angle)
            
        im0_6dRepNet = im0.copy()
        
        
        trans_vertices = renderer.transform_vertices(im0, global_poses)
        im0 = renderer.render(im0, trans_vertices, alpha=1.0)
        for i, [x1, y1, x2, y2] in enumerate(bboxes):
            # im0 = cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), 
                # [255,255,255], thickness=args.thickness)
            # im0 = cv2.putText(im0, str(round(scores[i], 3)), (int(x1), int(y1)), 
                # cv2.FONT_HERSHEY_PLAIN, 0.7, (255,255,255), thickness=2)
            [pitch, yaw, roll] = euler_angles[i]
            im0 = plot_3axis_Zaxis(im0, yaw, pitch, roll, tdx=(x1+x2)/2, tdy=(y1+y2)/2, 
                size=max(y2-y1, x2-x1)*0.75, thickness=args.thickness, extending=False)
        cv2.imwrite(single_path[:-4]+"_vis3d_res.jpg", im0)
        
        
        trans_vertices = renderer.transform_vertices(im0_6dRepNet, global_poses_6dRepNet)
        im0_6dRepNet = renderer.render(im0_6dRepNet, trans_vertices, alpha=1.0)
        for i, [x1, y1, x2, y2] in enumerate(bboxes):
            [pitch, yaw, roll] = euler_angles_6dRepNet[i]
            im0_6dRepNet = plot_3axis_Zaxis(im0_6dRepNet, yaw, pitch, roll, tdx=(x1+x2)/2, tdy=(y1+y2)/2, 
                size=max(y2-y1, x2-x1)*0.75, thickness=args.thickness, extending=False)
        cv2.imwrite(single_path[:-4]+"_vis3d_res_6dRepNet.jpg", im0_6dRepNet)
  