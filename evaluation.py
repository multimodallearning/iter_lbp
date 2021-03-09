import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

def jacobian_determinant(disp):
    device = disp.device
    
    gradz = nn.Conv3d(3, 3, (3, 1, 1), padding=(1, 0, 0), bias=False, groups=3)
    gradz.weight.data[:, 0, :, 0, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradz.to(device)
    grady = nn.Conv3d(3, 3, (1, 3, 1), padding=(0, 1, 0), bias=False, groups=3)
    grady.weight.data[:, 0, 0, :, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    grady.to(device)
    gradx = nn.Conv3d(3, 3, (1, 1, 3), padding=(0, 0, 1), bias=False, groups=3)
    gradx.weight.data[:, 0, 0, 0, :] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradx.to(device)
    
    disp = disp.permute(0, 4, 1, 2, 3)
    jacobian = torch.cat((gradz(disp), grady(disp), gradx(disp)), 0) + torch.eye(3, 3, device=device).view(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
        
    return jacdet.unsqueeze(0).unsqueeze(0)

def dice_coeff(outputs, labels, max_label):
    dice = torch.zeros(max_label)
    for label in range(1, max_label+1):
        iflat = (outputs==label).reshape(-1).float()
        tflat = (labels==label).reshape(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice

def copdgene_dists(data_dir, case_num, flow):
    device = flow.device
    _, D, H, W, _ = flow.shape
    
    lms_inh_path = os.path.join(data_dir, 'copd{}_300_iBH_xyz_r1.txt'.format(case_num))
    lms_inh = torch.cat((torch.tensor(np.loadtxt(lms_inh_path).astype('float32'), device=device) - 1, torch.ones((300, 1), device=device)), dim=1)
    lms_exh_path = os.path.join(data_dir, 'copd{}_300_eBH_xyz_r1.txt'.format(case_num))
    lms_exh = torch.cat((torch.tensor(np.loadtxt(lms_exh_path).astype('float32'), device=device) - 1, torch.ones((300, 1), device=device)), dim=1)

    shapes = torch.tensor([[250, 214, 303],
                           [285, 215, 240],
                           [287, 208, 295],
                           [257, 183, 285],
                           [269, 220, 293],
                           [256, 172, 258],
                           [266, 181, 268],
                           [255, 183, 275],
                           [290, 205, 265],
                           [321, 227, 265]], device=device)
    D_orig, H_orig, W_orig = shapes[case_num-1]
    
    voxel_spacing_inh = torch.tensor([[0.625, 0.645, 0.652, 0.590, 0.647, 0.633, 0.625, 0.586, 0.664, 0.742],
                                      [0.625, 0.645, 0.652, 0.590, 0.647, 0.633, 0.625, 0.586, 0.664, 0.742],
                                      [2.500, 2.500, 2.500, 2.500, 2.500, 2.500, 2.500, 2.500, 2.500, 2.500]], device=device).t()[case_num-1]

    crop = torch.tensor([ 55, 103,   1, 454, 444, 121,  69, 156,   1, 440, 449, 112,
                          38, 103,   3, 479, 436,  98,  44, 120,   5, 472, 440,  95,
                          33, 102,   3, 472, 420, 120,  46, 107,   4, 466, 420, 108,
                          34,  98,   3, 469, 408, 116,  55, 143,   3, 438, 413,  90,
                          44,  73,   9, 459, 412, 125,  63, 121,  11, 440, 413, 105,
                          52, 117,   5, 455, 387, 107,  66, 151,   8, 439, 402,  98,
                          46, 115,   6, 470, 403, 112,  56, 127,   3, 466, 404, 101,
                          39, 107,   6, 473, 419, 115,  56, 125,   6, 456, 414,  97,
                          33, 103,   5, 469, 410, 110,  51, 113,   6, 458, 411,  93,
                          35,  95,   7, 467, 400, 112,  57, 104,  13, 457, 401, 101], device=device).view(10,2,2,3)
    crop_inh = crop[case_num-1,0,:,:] - 1
    crop_exh = crop[case_num-1,1,:,:] - 1

    ST1 = torch.eye(4, device=device)
    ST1[:3, 3] = crop_inh[0, :]
    ST1[torch.arange(3),torch.arange(3)] = 1/voxel_spacing_inh
    new_size = torch.round((crop_inh[1, :] - crop_inh[0, :] + 1) * voxel_spacing_inh)

    voxel_spacing_exh = new_size / (crop_exh[1, :] - crop_exh[0, :] + 1)
    ST2 = torch.eye(4, device=device)
    ST2[:3, 3] = crop_exh[0, :]
    ST2[torch.arange(3),torch.arange(3)] = 1/voxel_spacing_exh

    lms_inh_ = torch.matmul(ST1.inverse(), lms_inh.t()).t()
    lms_exh_ = torch.matmul(ST2.inverse(), lms_exh.t()).t()

    disp = F.grid_sample(flow_world(flow.view(1, -1, 3), (D_orig,H_orig,W_orig)).view(1, D, H, W, 3).permute(0, 4, 1, 2, 3), kpts_pt(lms_inh_[:,:3].unsqueeze(0), (D_orig, H_orig, W_orig)).view(1, 1, 1, -1, 3)).view(3, -1).t()

    lms_exh_est_ = torch.cat((lms_inh_[:, :3] + disp, torch.ones((300, 1), device=device)), dim=1)
    lms_exh_est = torch.matmul(ST2, lms_exh_est_.t()).t()
    
    dists = torch.sqrt(torch.sum((voxel_spacing_inh * torch.round(lms_exh_est[:, :3] - lms_exh[:, :3])) ** 2, dim=1))
    
    return dists