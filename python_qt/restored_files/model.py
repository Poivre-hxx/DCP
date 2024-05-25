# Source Generated with Decompyle++
# File: model.cpython-39.pyc (Python 3.9)

import os
import sys
import copy
import math
import numpy as np
import torch
from torch.nn import nn
import torch.nn.functional
F = functional
nn
from torch.autograd import Variable

def clones(module, N):
    return None((lambda .0 = None: [ copy.deepcopy(module) for _ in .0 ])(range(N)))


def attention(query, key, value, mask, dropout = (None, None)):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e+09)
    p_attn = F.softmax(scores, -1, **('dim',))
    return (torch.matmul(p_attn, value), p_attn)


def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)
    distances = -torch.sum(src ** 2, 0, True, **('dim', 'keepdim')).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2, 0, True, **('dim', 'keepdim'))
    (distances, indices) = distances.topk(1, -1, **('k', 'dim'))
    return (distances, indices)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, 1, True, **('dim', 'keepdim'))
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k, -1, **('k', 'dim'))[1]
    return idx


def get_graph_feature(x, k = (20,)):
    idx = knn(x, k, **('k',))
    (batch_size, num_points, _) = idx.size()
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device, **('device',)).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    (_, num_dims, _) = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[(idx, :)]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature, x), 3, **('dim',)).permute(0, 3, 1, 2)
    return feature


class EncoderDecoder(nn.Module):
    '''
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    '''
    
    def __init__(self = None, encoder = None, decoder = None, src_embed = None, tgt_embed = None, generator = None):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    
    def forward(self, src, tgt, src_mask, tgt_mask):
        '''Take in and process masked src and target sequences.'''
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))

    __classcell__ = None


class Generator(nn.Module):
    
    def __init__(self = None, emb_dims = None):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(emb_dims, emb_dims // 2), nn.BatchNorm1d(emb_dims // 2), nn.ReLU(), nn.Linear(emb_dims // 2, emb_dims // 4), nn.BatchNorm1d(emb_dims // 4), nn.ReLU(), nn.Linear(emb_dims // 4, emb_dims // 8), nn.BatchNorm1d(emb_dims // 8), nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    
    def forward(self, x):
        x = self.nn(x.max(1, **('dim',))[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, 2, 1, True, **('p', 'dim', 'keepdim'))
        return (rotation, translation)

    __classcell__ = None


class Encoder(nn.Module):
    
    def __init__(self = None, layer = None, N = None):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    __classcell__ = None


class Decoder(nn.Module):
    '''Generic N layer decoder with masking.'''
    
    def __init__(self = None, layer = None, N = None):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

    __classcell__ = None


class LayerNorm(nn.Module):
    
    def __init__(self = None, features = None, eps = None):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    
    def forward(self, x):
        mean = x.mean(-1, True, **('keepdim',))
        std = x.std(-1, True, **('keepdim',))
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

    __classcell__ = None


class SublayerConnection(nn.Module):
    
    def __init__(self = None, size = None, dropout = None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    
    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))

    __classcell__ = None


class EncoderLayer(nn.Module):
    
    def __init__(self = None, size = None, self_attn = None, feed_forward = None, dropout = None):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    
    def forward(self, x, mask):
        x = None(None, (lambda x = None: self.self_attn(x, x, x, mask)))
        return self.sublayer[1](x, self.feed_forward)

    __classcell__ = None


class DecoderLayer(nn.Module):
    '''Decoder is made of self-attn, src-attn, and feed forward (defined below)'''
    
    def __init__(self = None, size = None, self_attn = None, src_attn = None, feed_forward = None, dropout = None):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    
    def forward(self, x, memory, src_mask, tgt_mask):
        '''Follow Figure 1 (right) for connections.'''
        m = memory
        x = None(None, (lambda x = None: self.self_attn(x, x, x, tgt_mask)))
        x = None(None, (lambda x = None: self.src_attn(x, m, m, src_mask)))
        return self.sublayer[2](x, self.feed_forward)

    __classcell__ = None


class MultiHeadedAttention(nn.Module):
    
    def __init__(self = None, h = None, d_model = None, dropout = None):
        '''Take in model size and number of heads.'''
        super(MultiHeadedAttention, self).__init__()
    # WARNING: Decompyle incomplete

    
    def forward(self, query, key, value, mask = (None,)):
        '''Implements Figure 2'''
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        (query, key, value) = (lambda .0 = None: [ l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous() for l, x in .0 ])(zip(self.linears, (query, key, value)))
        (x, self.attn) = attention(query, key, value, mask, self.dropout, **('mask', 'dropout'))
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    __classcell__ = None


class PositionwiseFeedForward(nn.Module):
    '''Implements FFN equation.'''
    
    def __init__(self = None, d_model = None, d_ff = None, dropout = None):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    
    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())

    __classcell__ = None


class PointNet(nn.Module):
    
    def __init__(self = None, emb_dims = None):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1, False, **('kernel_size', 'bias'))
        self.conv2 = nn.Conv1d(64, 64, 1, False, **('kernel_size', 'bias'))
        self.conv3 = nn.Conv1d(64, 64, 1, False, **('kernel_size', 'bias'))
        self.conv4 = nn.Conv1d(64, 128, 1, False, **('kernel_size', 'bias'))
        self.conv5 = nn.Conv1d(128, emb_dims, 1, False, **('kernel_size', 'bias'))
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x

    __classcell__ = None


class DGCNN(nn.Module):
    
    def __init__(self = None, emb_dims = None):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, 1, False, **('kernel_size', 'bias'))
        self.conv2 = nn.Conv2d(64, 64, 1, False, **('kernel_size', 'bias'))
        self.conv3 = nn.Conv2d(64, 128, 1, False, **('kernel_size', 'bias'))
        self.conv4 = nn.Conv2d(128, 256, 1, False, **('kernel_size', 'bias'))
        self.conv5 = nn.Conv2d(512, emb_dims, 1, False, **('kernel_size', 'bias'))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    
    def forward(self, x):
        (batch_size, num_dims, num_points) = x.size()
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(-1, True, **('dim', 'keepdim'))[0]
        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(-1, True, **('dim', 'keepdim'))[0]
        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(-1, True, **('dim', 'keepdim'))[0]
        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(-1, True, **('dim', 'keepdim'))[0]
        x = torch.cat((x1, x2, x3, x4), 1, **('dim',))
        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x

    __classcell__ = None


class MLPHead(nn.Module):
    
    def __init__(self = None, args = None):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dims
        self.emb_dims = emb_dims
        self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2), nn.BatchNorm1d(emb_dims // 2), nn.ReLU(), nn.Linear(emb_dims // 2, emb_dims // 4), nn.BatchNorm1d(emb_dims // 4), nn.ReLU(), nn.Linear(emb_dims // 4, emb_dims // 8), nn.BatchNorm1d(emb_dims // 8), nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    __classcell__ = None


class Identity(nn.Module):
    
    def __init__(self = None):
        super(Identity, self).__init__()

    
    def forward(self, *input):
        return input

    __classcell__ = None


class Transformer(nn.Module):
    
    def __init__(self = None, args = None):
        super(Transformer, self).__init__()
        self.emb_dims = args.emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N), Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N), nn.Sequential(), nn.Sequential(), nn.Sequential())

    
    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return (src_embedding, tgt_embedding)

    __classcell__ = None


class SVDHead(nn.Module):
    
    def __init__(self = None, args = None):
        super(SVDHead, self).__init__()
        self.emb_dims = args.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), False, **('requires_grad',))
        self.reflect[(2, 2)] = -1

    
    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size = src.size(0)
        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, 2, **('dim',))
        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())
        src_centered = src - src.mean(2, True, **('dim', 'keepdim'))
        src_corr_centered = src_corr - src_corr.mean(2, True, **('dim', 'keepdim'))
        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())
        U = []
        S = []
        V = []
        R = []
        for i in range(src.size(0)):
            (u, s, v) = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                (u, s, v) = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
            R.append(r)
            U.append(u)
            S.append(s)
            V.append(v)
        U = torch.stack(U, 0, **('dim',))
        V = torch.stack(V, 0, **('dim',))
        S = torch.stack(S, 0, **('dim',))
        R = torch.stack(R, 0, **('dim',))
        t = torch.matmul(-R, src.mean(2, True, **('dim', 'keepdim'))) + src_corr.mean(2, True, **('dim', 'keepdim'))
        return (R, t.view(batch_size, 3))

    __classcell__ = None


class DCP(nn.Module):
    
    def __init__(self = None, args = None):
        super(DCP, self).__init__()
        self.emb_dims = args.emb_dims
        self.cycle = args.cycle
        if args.emb_nn == 'pointnet':
            self.emb_nn = PointNet(self.emb_dims, **('emb_dims',))
        elif args.emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(self.emb_dims, **('emb_dims',))
        else:
            raise Exception('Not implemented')
        if None.pointer == 'identity':
            self.pointer = Identity()
        elif args.pointer == 'transformer':
            self.pointer = Transformer(args, **('args',))
        else:
            raise Exception('Not implemented')
        if None.head == 'mlp':
            self.head = MLPHead(args, **('args',))
        elif args.head == 'svd':
            self.head = SVDHead(args, **('args',))
        else:
            raise Exception('Not implemented')

    
    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)
        (src_embedding_p, tgt_embedding_p) = self.pointer(src_embedding, tgt_embedding)
        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p
        (rotation_ab, translation_ab) = self.head(src_embedding, tgt_embedding, src, tgt)
        if self.cycle:
            (rotation_ba, translation_ba) = self.head(tgt_embedding, src_embedding, tgt, src)
        else:
            rotation_ba = rotation_ab.transpose(2, 1).contiguous()
            translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)
        return (rotation_ab, translation_ab, rotation_ba, translation_ba)

    __classcell__ = None

