# Source Generated with Decompyle++
# File: dcp_net.cpython-39.pyc (Python 3.9)

import numpy as np
import torch
import torch
import os
from model import DCP
import argparse
import open3d as o3d
import copy

def run_one_pointcloud(src, target, net):
    if len(src.shape) == 2 and len(target.shape) == 2:
        src = np.expand_dims(src[(:, :3)], 0, **('axis',))
        src = np.transpose(src, [
            0,
            2,
            1])
        target = np.expand_dims(target[(:, :3)], 0, **('axis',))
        target = np.transpose(target, [
            0,
            2,
            1])
    net.eval()
    src = torch.from_numpy(src).cuda()
    target = torch.from_numpy(target).cuda()
    src = src.float()
    target = target.float()
    (rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred) = net(src, target)
    rotation_ab_pred = rotation_ab_pred.detach().cpu().numpy()
    translation_ab_pred = translation_ab_pred.detach().cpu().numpy()
    rotation_ba_pred = rotation_ba_pred.detach().cpu().numpy()
    translation_ba_pred = translation_ba_pred.detach().cpu().numpy()
    return (rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred)


def use_dcp(src, tgt, model_path):
    parser = argparse.ArgumentParser('Point Cloud Registration', **('description',))
    parser.add_argument('--exp_name', str, '', 'N', 'Name of the experiment', **('type', 'default', 'metavar', 'help'))
    parser.add_argument('--model', str, 'dcp', 'N', [
        'dcp'], 'Model to use, [dcp]', **('type', 'default', 'metavar', 'choices', 'help'))
    parser.add_argument('--emb_nn', str, 'dgcnn', 'N', [
        'pointnet',
        'dgcnn'], 'Embedding nn to use, [pointnet, dgcnn]', **('type', 'default', 'metavar', 'choices', 'help'))
    parser.add_argument('--pointer', str, 'transformer', 'N', [
        'identity',
        'transformer'], 'Attention-based pointer generator to use, [identity, transformer]', **('type', 'default', 'metavar', 'choices', 'help'))
    parser.add_argument('--head', str, 'svd', 'N', [
        'mlp',
        'svd'], 'Head to use, [mlp, svd]', **('type', 'default', 'metavar', 'choices', 'help'))
    parser.add_argument('--emb_dims', int, 512, 'N', 'Dimension of embeddings', **('type', 'default', 'metavar', 'help'))
    parser.add_argument('--n_blocks', int, 1, 'N', 'Num of blocks of encoder&decoder', **('type', 'default', 'metavar', 'help'))
    parser.add_argument('--n_heads', int, 4, 'N', 'Num of heads in multiheadedattention', **('type', 'default', 'metavar', 'help'))
    parser.add_argument('--ff_dims', int, 1024, 'N', 'Num of dimensions of fc in transformer', **('type', 'default', 'metavar', 'help'))
    parser.add_argument('--dropout', float, 0, 'N', 'Dropout ratio in transformer', **('type', 'default', 'metavar', 'help'))
    parser.add_argument('--batch_size', int, 32, 'batch_size', 'Size of batch)', **('type', 'default', 'metavar', 'help'))
    parser.add_argument('--test_batch_size', int, 1, 'batch_size', 'Size of batch)', **('type', 'default', 'metavar', 'help'))
    parser.add_argument('--epochs', int, 250, 'N', 'number of episode to train ', **('type', 'default', 'metavar', 'help'))
    parser.add_argument('--use_sgd', 'store_true', False, 'Use SGD', **('action', 'default', 'help'))
    parser.add_argument('--lr', float, 0.001, 'LR', 'learning rate (default: 0.001, 0.1 if using sgd)', **('type', 'default', 'metavar', 'help'))
    parser.add_argument('--momentum', float, 0.9, 'M', 'SGD momentum (default: 0.9)', **('type', 'default', 'metavar', 'help'))
    parser.add_argument('--no_cuda', 'store_true', False, 'enables CUDA training', **('action', 'default', 'help'))
    parser.add_argument('--seed', int, 1234, 'S', 'random seed (default: 1)', **('type', 'default', 'metavar', 'help'))
    parser.add_argument('--eval', 'store_true', False, 'evaluate the model', **('action', 'default', 'help'))
    parser.add_argument('--cycle', bool, False, 'N', 'Whether to use cycle consistency', **('type', 'default', 'metavar', 'help'))
    parser.add_argument('--model_path', str, 'dcp_v2.t7', 'N', ' model path', **('type', 'default', 'metavar', 'help'))
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    net = DCP(args).cuda()
    net.load_state_dict(torch.load(model_path), False, **('strict',))
    point1 = np.asarray(src.points)
    point2 = np.asarray(tgt.points)
    (r_ab, t_ab, r_ba, t_ba) = run_one_pointcloud(point1, point2, net)
    T = np.array([
        [
            r_ab[0][0][0],
            r_ab[0][0][1],
            r_ab[0][0][2],
            t_ab[0][0]],
        [
            r_ab[0][1][0],
            r_ab[0][1][1],
            r_ab[0][1][2],
            t_ab[0][1]],
        [
            r_ab[0][2][0],
            r_ab[0][2][1],
            r_ab[0][2][2],
            t_ab[0][2]],
        [
            0,
            0,
            0,
            1]])
    return T

if __name__ == '__main__':
    index = 0
    src = o3d.io.read_point_cloud('airplane.ply')
    tgt = o3d.io.read_point_cloud('airplane1.ply')
    T = use_dcp(src, tgt, 'dcp_v2.t7')
    trans_cloud = copy.deepcopy(src)
    trans_cloud.transform(T)
    src.paint_uniform_color([
        1,
        0,
        0])
    tgt.paint_uniform_color([
        0,
        1,
        0])
    trans_cloud.paint_uniform_color([
        0,
        0,
        1])
    o3d.visualization.draw_geometries([
        src,
        tgt,
        trans_cloud], 800, **('width',))
    print('finished.')
