import numpy as np
import torch
import torch.nn.functional as F

##### misc #####

def parameter_count(model):
    print('# Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

def gpu_usage():
    print('GPU usage (current/max): {:.2f} / {:.2f} GB'.format(torch.cuda.memory_allocated()*1e-9, torch.cuda.max_memory_allocated()*1e-9))
    
##### kpts / graph #####

def kpts_pt(kpts_world, shape, align_corners=None):
    dtype = kpts_world.dtype
    device = kpts_world.device
    D, H, W = shape
   
    kpts_pt_ = (kpts_world.flip(-1) / (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)) * 2 - 1
    if not align_corners:
        kpts_pt_ *= (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)/torch.tensor([W, H, D], dtype=dtype, device=device)
    
    return kpts_pt_

def kpts_world(kpts_pt, shape, align_corners=None):
    dtype = kpts_pt.dtype
    device = kpts_pt.device
    D, H, W = shape
    
    if not align_corners:
        kpts_pt /= (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)/torch.tensor([W, H, D], dtype=dtype, device=device)
    kpts_world_ = (((kpts_pt + 1) / 2) * (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)).flip(-1) 
    
    return kpts_world_

def flow_pt(flow_world, shape, align_corners=None):
    dtype = flow_world.dtype
    device = flow_world.device
    D, H, W = shape
    
    flow_pt_ = (flow_world.flip(-1) / (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)) * 2
    if not align_corners:
        flow_pt_ *= (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)/torch.tensor([W, H, D], dtype=dtype, device=device)
    
    return flow_pt_

def flow_world(flow_pt, shape, align_corners=None):
    dtype = flow_pt.dtype
    device = flow_pt.device
    D, H, W = shape
    
    if not align_corners:
        flow_pt /= (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)/torch.tensor([W, H, D], dtype=dtype, device=device)
    flow_world_ = ((flow_pt / 2) * (torch.tensor([W, H, D], dtype=dtype, device=device) - 1)).flip(-1)
    
    return flow_world_

def random_kpts(mask, d, num_points=None):
    _, _, D, H, W = mask.shape
    device = mask.device
    
    kpts = torch.nonzero(mask[:, :, ::d, ::d, ::d]).unsqueeze(0).float()[:, :, 2:]
    
    if not num_points is None:
        kpts = kpts[:, torch.randperm(kpts.shape[1])[:num_points], :]
    
    return kpts_pt(kpts, (D//d, H//d, W//d))

def knn_graph(kpts, k, include_self=False):
    B, N, D = kpts.shape
    device = kpts.device
    
    dist = pdist(kpts)
    ind = (-dist).topk(k + (1 - int(include_self)), dim=-1)[1][:, :, 1 - int(include_self):]
    A = torch.zeros(B, N, N).to(device)
    A[:, torch.arange(N).repeat(k), ind[0].t().contiguous().view(-1)] = 1
    A[:, ind[0].t().contiguous().view(-1), torch.arange(N).repeat(k)] = 1
    
    return ind, dist*A, A

def lbp_graph(kpts_fixed, k):
    device = kpts_fixed.device
    
    A = knn_graph(kpts_fixed, k, include_self=False)[2][0]
    edges = A.nonzero()
    edges_idx = torch.zeros_like(A).long()
    edges_idx[A.bool()] = torch.arange(edges.shape[0]).to(device)
    edges_reverse_idx = edges_idx.t()[A.bool()]
    
    return edges, edges_reverse_idx

##### filter #####

def filter1D(img, weight, dim, padding_mode='replicate'):
    B, C, D, H, W = img.shape
    N = weight.shape[0]
    
    padding = torch.zeros(6,)
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N//2
    padding = padding.long().tolist()
    
    view = torch.ones(5,)
    view[dim + 2] = -1
    view = view.long().tolist()
    
    return F.conv3d(F.pad(img.view(B*C, 1, D, H, W), padding, mode=padding_mode), weight.view(view)).view(B, C, D, H, W)

def smooth(img, sigma):
    device = img.device
    
    sigma = torch.tensor([sigma]).to(device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1
    
    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N).to(device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()
    
    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)
    
    return img

def mean_filter(img, r):
    device = img.device
    
    weight = torch.ones((2*r+1,), device=device)/(2*r+1)
    
    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)
    
    return img

##### distance #####

def pdist(x, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=3)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist

def pdist2(x, y, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(dim=3)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = (y**2).sum(dim=2).unsqueeze(1)
        dist = xx + yy - 2.0 * torch.bmm(x, y.permute(0, 2, 1))
    return dist

def ssd(kpts_fixed, feat_fixed, feat_moving, orig_shape, disp_radius=16, disp_step=2, patch_radius=3, alpha=1, unroll_factor=100):
    device = kpts_fixed.device
    N = kpts_fixed.shape[1]
    C = feat_fixed.shape[1]
    D, H, W = orig_shape
    
    patch_step = disp_step # same stride necessary for fast implementation
    patch = torch.stack(torch.meshgrid(torch.arange(0, 2 * patch_radius + 1, patch_step, device=device),
                                       torch.arange(0, 2 * patch_radius + 1, patch_step, device=device),
                                       torch.arange(0, 2 * patch_radius + 1, patch_step, device=device)), dim=3).view(1, -1, 3) - patch_radius
    patch = flow_pt(patch, (D, H, W), align_corners=True).view(1, 1, -1, 1, 3)
    
    patch_width = round(patch.shape[2] ** (1.0 / 3))
    
    if patch_width % 2 == 0:
        pad = [(patch_width - 1) // 2, (patch_width - 1) // 2 + 1]
    else:
        pad = [(patch_width - 1) // 2, (patch_width - 1) // 2]
    
    disp = torch.stack(torch.meshgrid(torch.arange(- disp_step * (disp_radius + ((pad[0] + pad[1]) / 2)), (disp_step * (disp_radius + ((pad[0] + pad[1]) / 2))) + 1, disp_step, device=device),
                                      torch.arange(- disp_step * (disp_radius + ((pad[0] + pad[1]) / 2)), (disp_step * (disp_radius + ((pad[0] + pad[1]) / 2))) + 1, disp_step, device=device),
                                      torch.arange(- disp_step * (disp_radius + ((pad[0] + pad[1]) / 2)), (disp_step * (disp_radius + ((pad[0] + pad[1]) / 2))) + 1, disp_step, device=device)), dim=3).view(1, -1, 3)
    disp = flow_pt(disp, (D, H, W), align_corners=True).view(1, 1, -1, 1, 3)
    
    disp_width = disp_radius * 2 + 1
    
    cost = torch.zeros((1, N, disp_width, disp_width, disp_width), device=device)
    split = np.array_split(np.arange(N), unroll_factor)
    for i in range(unroll_factor):
        feat_fixed_patch = F.grid_sample(feat_fixed, kpts_fixed[:, split[i], :].view(1, -1, 1, 1, 3) + patch, padding_mode='border', align_corners=True)
        feat_moving_disp = F.grid_sample(feat_moving, kpts_fixed[:, split[i], :].view(1, -1, 1, 1, 3) + disp, padding_mode='border', align_corners=True)        
        corr = F.conv3d(feat_moving_disp.view(1, -1, disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1]), feat_fixed_patch.view(-1, 1, patch_width, patch_width, patch_width), groups = C * split[i].shape[0]).view(C, split[i].shape[0], disp_width, disp_width, disp_width)
        patch_sum = (feat_fixed_patch ** 2).sum(dim=3).view(C, split[i].shape[0], 1, 1, 1)
        disp_sum = (patch_width ** 3) * F.avg_pool3d((feat_moving_disp ** 2).view(C, -1, disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1]), patch_width, stride=1).view(C, split[i].shape[0], disp_width, disp_width, disp_width)
        cost[0, split[i], :, :, :] = ((- 2 * corr + patch_sum + disp_sum)).sum(0)
    
    cost *= (alpha/(patch_width ** 3))
    
    return cost

##### message passing #####

def minconv(input):
    device = input.device
    disp_width = input.shape[-1]
    
    disp1d = torch.linspace(-(disp_width//2), disp_width//2, disp_width, device=device)
    regular1d = (disp1d.view(1,-1) - disp1d.view(-1,1)) ** 2
    
    output = torch.min( input.view(-1, disp_width, 1, disp_width, disp_width) + regular1d.view(1, disp_width, disp_width, 1, 1), 1)[0]
    output = torch.min(output.view(-1, disp_width, disp_width, 1, disp_width) + regular1d.view(1, 1, disp_width, disp_width, 1), 2)[0]
    output = torch.min(output.view(-1, disp_width, disp_width, disp_width, 1) + regular1d.view(1, 1, 1, disp_width, disp_width), 3)[0]
    
    output = output - (torch.min(output.view(-1, disp_width**3), 1)[0]).view(-1, 1, 1, 1)

    return output.view_as(input)

def sparse_minconv(multi_data_cost, candidates_edges0, candidates_edges1):
    regularised = torch.min(multi_data_cost.unsqueeze(1) + (candidates_edges0.unsqueeze(1) - candidates_edges1.unsqueeze(2)).pow(2).sum(3), 2)[0]
    return regularised