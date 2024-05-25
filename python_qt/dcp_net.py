# -*- coding: utf-8 -*-

import numpy as np

import torch

import os

from model import DCP

import argparse

import open3d as o3d

import copy



def run_one_pointcloud(src, target, net):

    if len(src.shape) == 2 and len(target.shape) == 2:

        src = np.expand_dims(src[:, :3], axis=0)

        src = np.transpose(src, [0, 2, 1])

        target = np.expand_dims(target[:, :3], axis=0)

        target = np.transpose(target, [0, 2, 1])

    net.eval()

    src = torch.from_numpy(src).cuda()

    target = torch.from_numpy(target).cuda()

    src = src.float()

    target = target.float()

    rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)

    rotation_ab_pred = rotation_ab_pred.detach().cpu().numpy()

    translation_ab_pred = translation_ab_pred.detach().cpu().numpy()

    rotation_ba_pred = rotation_ba_pred.detach().cpu().numpy()

    translation_ba_pred = translation_ba_pred.detach().cpu().numpy()

    return rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred



def use_dcp(src, tgt, model_path):

    parser = argparse.ArgumentParser(description='Point Cloud Registration')

    parser.add_argument('--exp_name', type=str, default='', metavar='N', help='Name of the experiment')

    parser.add_argument('--model', type=str, default='dcp', choices=['dcp'], metavar='N', help='Model to use, [dcp]')

    parser.add_argument('--emb_nn', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], metavar='N', help='Embedding nn to use, [pointnet, dgcnn]')

    parser.add_argument('--pointer', type=str, default='transformer', choices=['identity', 'transformer'], metavar='N', help='Attention-based pointer generator to use, [identity, transformer]')

    parser.add_argument('--head', type=str, default='svd', choices=['mlp', 'svd'], metavar='N', help='Head to use, [mlp, svd]')

    parser.add_argument('--emb_dims', type=int, default=512, metavar='N', help='Dimension of embeddings')

    parser.add_argument('--n_blocks', type=int, default=1, metavar='N', help='Num of blocks of encoder&decoder')

    parser.add_argument('--n_heads', type=int, default=4, metavar='N', help='Num of heads in multiheadedattention')

    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N', help='Num of dimensions of fc in transformer')

    parser.add_argument('--dropout', type=float, default=0, metavar='N', help='Dropout ratio in transformer')

    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of batch)')

    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch)')

    parser.add_argument('--epochs', type=int, default=250, metavar='N', help='number of episode to train ')

    parser.add_argument('--use_sgd', action='store_true', default=False, help='Use SGD')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')

    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--eval', action='store_true', default=False, help='evaluate the model')

    parser.add_argument('--cycle', type=bool, default=False, metavar='N', help='Whether to use cycle consistency')

    parser.add_argument('--model_path', type=str, default='dcp_v2.t7', metavar='N', help=' model path')

    args = parser.parse_args()



    torch.backends.cudnn.deterministic = True

    torch.manual_seed(args.seed)

    torch.cuda.manual_seed_all(args.seed)

    net = DCP(args).cuda()

    net.load_state_dict(torch.load(model_path), strict=False)



    point1 = np.asarray(src.points)

    point2 = np.asarray(tgt.points)

    r_ab, t_ab, r_ba, t_ba = run_one_pointcloud(point1, point2, net)



    T = np.array([

        [r_ab[0][0][0], r_ab[0][0][1], r_ab[0][0][2], t_ab[0][0]],

        [r_ab[0][1][0], r_ab[0][1][1], r_ab[0][1][2], t_ab[0][1]],

        [r_ab[0][2][0], r_ab[0][2][1], r_ab[0][2][2], t_ab[0][2]],

        [0, 0, 0, 1]

    ])

    return T



if __name__ == '__main__':

    src = o3d.io.read_point_cloud('airplane.ply')

    tgt = o3d.io.read_point_cloud('airplane1.ply')

    T = use_dcp(src, tgt, 'dcp_v2.t7')

    trans_cloud = copy.deepcopy(src)

    trans_cloud.transform(T)

    src.paint_uniform_color([1, 0, 0])

    tgt.paint_uniform_color([0, 1, 0])

    trans_cloud.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([src, tgt, trans_cloud], width=800)

    print('finished.')

