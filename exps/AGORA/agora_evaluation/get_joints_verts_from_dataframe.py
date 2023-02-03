#------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
#------------------------------------------------------------------------------
import argparse
import logging
import os
from tqdm import tqdm

import numpy as np
import pandas
import pickle
import torch

from .projection import project_2d
from .utils import load_model


logging.basicConfig(level=logging.DEBUG)


def get_smpl_vertices(
        kid_flag,
        gt,
        gender,
        smpl_neutral_kid,
        smpl_neutral,
        pose2rot=True):
    # All neutral model are used
    if kid_flag:
        model_gt = smpl_neutral_kid
        num_betas = 11
    else:

        model_gt = smpl_neutral
        num_betas = 10

    # Since SMPLX to SMPL conversion tool store root_pose and translation as
    # keys
    if 'root_pose' in gt.keys():
        gt['global_orient'] = gt.pop('root_pose')
    if 'translation' in gt.keys():
        gt['transl'] = gt.pop('translation')
    for k, v in gt.items():
        if torch.is_tensor(v):
            gt[k] = v.detach().cpu().numpy()

    smplx_gt = model_gt(betas=torch.tensor(gt['betas'][:, :num_betas], dtype=torch.float),
                        global_orient=torch.tensor(gt['global_orient'], dtype=torch.float),
                        body_pose=torch.tensor(gt['body_pose'], dtype=torch.float),
                        transl=torch.tensor(gt['transl'], dtype=torch.float), pose2rot=pose2rot)

    return smplx_gt.joints.detach().cpu().numpy().squeeze(
    ), smplx_gt.vertices.detach().cpu().numpy().squeeze()


def get_smplx_vertices(
        num_betas,
        kid_flag,
        gt,
        gender,
        smplx_male_kid_gt,
        smplx_female_kid_gt,
        smplx_neutral_kid,
        smplx_male_gt,
        smplx_female_gt,
        smplx_neutral,
        pose2rot=True):

    if kid_flag:
        num_betas = 11
        if gender == 'female':
            model_gt = smplx_neutral_kid
        elif gender == 'male':
            model_gt = smplx_male_kid_gt
        elif gender == 'neutral':
            model_gt = smplx_neutral_kid
        else:
            raise KeyError(
                'Kid: Got gender {}, what gender is it?'.format(gender))

    else:
        if gender == 'female':
            model_gt = smplx_female_gt
        elif gender == 'male':
            model_gt = smplx_male_gt
        elif gender == 'neutral':
            model_gt = smplx_neutral
        else:
            raise KeyError('Got gender {}, what gender is it?'.format(gender))

    smplx_gt = model_gt(
        betas=torch.tensor(gt['betas'][:, :num_betas], dtype=torch.float), 
        global_orient=torch.tensor(gt['global_orient'], dtype=torch.float),
        body_pose=torch.tensor(gt['body_pose'], dtype=torch.float),
        left_hand_pose=torch.tensor(gt['left_hand_pose'], dtype=torch.float),
        right_hand_pose=torch.tensor(gt['right_hand_pose'], dtype=torch.float),
        transl=torch.tensor(gt['transl'], dtype=torch.float),
        expression=torch.tensor(gt['expression'], dtype=torch.float), 
        jaw_pose=torch.tensor(gt['jaw_pose'], dtype=torch.float),
        leye_pose=torch.tensor(gt['leye_pose'], dtype=torch.float),
        reye_pose=torch.tensor(gt['reye_pose'], dtype=torch.float), pose2rot=pose2rot)

    return smplx_gt.joints.detach().cpu().numpy().squeeze(
    ), smplx_gt.vertices.detach().cpu().numpy().squeeze()


def get_pred_joints(predJoints, predVerts, df, idx, pNum, args):
    # Add kid flag
    if 'kid_flag' in df.iloc[idx].at['pred'][pNum].keys():
        kid_flag = bool(df.iloc[idx].at['pred'][pNum]['kid_flag'])
    else:
        kid_flag = False

    # Add pose2rot if pose is rotation matrix
    if 'pose2rot' in df.iloc[idx].at['pred'][pNum].keys():
        args.pose2rot = bool(df.iloc[idx].at['pred'][pNum]['pose2rot'])
    else:
        args.pose2rot = True

    # Add gender
    if 'gender' in df.iloc[idx].at['pred'][pNum].keys():
        gender = df.iloc[idx].at['pred'][pNum]['gender']
    else:
        gender = 'neutral'

    # Add num_betas for shape
    if 'num_betas' in df.iloc[idx].at['pred'][pNum].keys():
        num_betas = df.iloc[idx].at['pred'][pNum]['num_betas']
    else:
        num_betas = 10

    if not 'model_male' in globals():
        args.numBetas = num_betas
        global model_male, model_male_kid, model_female, model_female_kid, model_neutral, model_neutral_kid
        model_male, model_male_kid, model_female, model_female_kid, model_neutral, model_neutral_kid = load_model(
            args)
    # Add 3d joints
    if 'allSmplJoints3d' in df.iloc[idx].at['pred'][pNum].keys():
        predJoints.append(df.iloc[idx].at['pred'][pNum]
                          ['allSmplJoints3d'].squeeze())
    elif 'params' in df.iloc[idx].at['pred'][pNum].keys():
        # Calculate joints and vertices from params
        gt = df.iloc[idx].at['pred'][pNum]['params']
        if args.modeltype == 'SMPLX':
            pred_joints, pred_verts = get_smplx_vertices(num_betas, kid_flag, gt, gender, model_male_kid,
                                                         model_female_kid, model_neutral_kid, model_male,
                                                         model_female, model_neutral, args.pose2rot)
        elif args.modeltype == 'SMPL':
            pred_joints, pred_verts = get_smpl_vertices(
                kid_flag, gt, gender, model_neutral_kid, model_neutral, args.pose2rot)
        else:
            logging.DEBUG('model ' + args.modeltype + ' not defined')
        predJoints.append(pred_joints)
    else:
        raise KeyError('no predicted paramaters are provided')

    # Add 3d verts
    if 'verts' in df.iloc[idx].at['pred'][pNum].keys():
        predVerts.append(df.iloc[idx].at['pred'][pNum]['verts'].squeeze())
    else:
        predVerts.append(pred_verts)


def get_projected_joints(
        args,
        df,
        idx,
        pNum,
        model_male_kid_gt,
        model_female_kid,
        model_neutral_kid,
        model_male_gt,
        model_female_gt,
        model_neutral,
        debug=False):
    kid_flag = df.iloc[idx].at['kid'][pNum]
    gender = df.iloc[idx].at['gender'][pNum]

    if args.modeltype == 'SMPL':
        smpl_path = os.path.join(
            args.gt_model_path,
            df.iloc[idx].at['gt_path_smpl'][pNum]).replace(
            '.obj',
            '.pkl')
        gt = pickle.load(open(smpl_path, 'rb'))
        gt_joints_local, gt_verts_local = get_smpl_vertices(
            kid_flag, gt, gender, model_neutral_kid, model_neutral)
    else:
        smplx_path = os.path.join(
            args.gt_model_path,
            df.iloc[idx].at['gt_path_smplx'][pNum]).replace(
            '.obj',
            '.pkl')
        gt = pickle.load(open(smplx_path, 'rb'))
        gt_joints_local, gt_verts_local = get_smplx_vertices(args.numBetas, kid_flag, gt, gender,
                                                             model_male_kid_gt, model_female_kid,
                                                             model_neutral_kid, model_male_gt,
                                                             model_female_gt, model_neutral)
    if debug:
        from psbody.mesh import Mesh
        Mesh(v=gt_verts_local).show()

    # gt_verts_cam_2d, gt_verts_cam_3d = project_2d(args, df, idx, pNum, gt_verts_local)

    # gt_joints_cam_2d, gt_joints_cam_3d = project_2d(args, df, idx, pNum, gt_joints_local)
    # return gt_verts_cam_2d, gt_verts_cam_3d, gt_joints_cam_2d, gt_joints_cam_3d
    
    gt_verts_cam_2d, gt_verts_cam_3d, _, _, _, _ = project_2d(args, df, idx, pNum, gt_verts_local)
    gt_joints_cam_2d, gt_joints_cam_3d, cam_j3d, camR, camT, camK = project_2d(args, df, idx, pNum, gt_joints_local)
    return gt_verts_cam_2d, gt_verts_cam_3d, gt_joints_cam_2d, gt_joints_cam_3d, cam_j3d, camR, camT, camK 


def add_joints_verts_in_dataframe(args, df, store_joints):

    model_male, model_male_kid, model_female, model_female_kid, model_neutral, model_neutral_kid = load_model(
        args)

    df.insert(column='gt_joints_2d', loc=len(df.columns), value=len(df) * [''])
    df.insert(column='gt_joints_3d', loc=len(df.columns), value=len(df) * [''])
    df.insert(column='gt_verts', loc=len(df.columns), value=len(df) * [''])
    df.insert(column='cam_j3d', loc=len(df.columns), value=len(df) * [''])  # newly added for Euler angles Calculation
    df.insert(column='camR', loc=len(df.columns), value=len(df) * [''])  # newly added for Euler angles Calculation
    df.insert(column='camT', loc=len(df.columns), value=len(df) * [''])  # newly added for Euler angles Calculation
    df.insert(column='camK', loc=len(df.columns), value=len(df) * [''])  # newly added for Euler angles Calculation
    
    for idx in tqdm(range(len(df))):
        for jdx in range(len(df.iloc[idx]['isValid'])):
            if jdx == 0:
                if store_joints:
                    df.iloc[idx, df.columns.get_loc('gt_joints_2d')] = []
                    df.iloc[idx, df.columns.get_loc('gt_joints_3d')] = []
                    df.iloc[idx, df.columns.get_loc('gt_verts')] = []
                    df.iloc[idx, df.columns.get_loc('cam_j3d')] = []
                    df.iloc[idx, df.columns.get_loc('camR')] = []
                    df.iloc[idx, df.columns.get_loc('camT')] = []
                    df.iloc[idx, df.columns.get_loc('camK')] = []

            if store_joints:
                # gt_verts_cam_2d, gt_verts_cam_3d, gt_joints_cam_2d, gt_joints_cam_3d = get_projected_joints(args, df,
                                                                                                            # idx, jdx,
                                                                                                            # model_male_kid,
                                                                                                            # model_female_kid,
                                                                                                            # model_neutral_kid,
                                                                                                            # model_male,
                                                                                                            # model_female,
                                                                                                            # model_neutral,
                                                                                                            # debug=False)
                gt_verts_cam_2d, gt_verts_cam_3d, gt_joints_cam_2d, gt_joints_cam_3d, cam_j3d, camR, camT, camK = get_projected_joints(args, df, idx, jdx, model_male_kid, model_female_kid, model_neutral_kid, model_male, model_female, model_neutral, debug=False)
                
                df.iloc[idx].at['gt_joints_2d'].append(gt_joints_cam_2d)
                df.iloc[idx].at['gt_joints_3d'].append(gt_joints_cam_3d)
                df.iloc[idx].at['gt_verts'].append(gt_verts_cam_3d)
                df.iloc[idx].at['cam_j3d'].append(cam_j3d)
                df.iloc[idx].at['camR'].append(camR)
                df.iloc[idx].at['camT'].append(camT)
                df.iloc[idx].at['camK'].append(camK)

    return df
