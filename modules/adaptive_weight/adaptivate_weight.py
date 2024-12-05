import torch
import torch.nn.functional as F
from math import sqrt
import cv2
import numpy as np

def compute_colorloss_weight(sample, device = "cuda"):
    H, W = sample["H"], sample["W"]
    pixels_xy = sample["rays_xy"].to(device)    # 在当前图像上采样的像素点坐标
    pixels_xy[:, 0] *= W
    pixels_xy[:, 1] *= H
    imgs_pairs = sample["imgs_pairs"].to(device)          # 当前图像与相邻几帧的图像
    poses_pairs = sample["poses_pairs"].to(device)     # 当前图像的位姿与相邻图像的位姿
    depths = sample["rendered_depth"].to(device).reshape(-1, 1)              # 像素采样点的深度
    normals = sample["rendered_normal"].to(device).reshape(-1, 1, 3)    # 像素采样点对应的世界坐标法向
    intrinsics = sample["intrinsic"].to(device)     # 相机内参
    intrinsics_inv = torch.linalg.inv(intrinsics)
    intrinsics = intrinsics[None, ...].expand(poses_pairs.shape[0], 3, 3)
    intrinsics_inv = intrinsics_inv[None, ...].expand(poses_pairs.shape[0], 3, 3)
    
    pixel_expand = torch.cat([pixels_xy, torch.ones_like(depths).to(pixels_xy.device)], dim = -1)
    uv = torch.matmul(intrinsics_inv[0],  pixel_expand.T).T
    project_xyz = (uv * depths).reshape(-1, 3, 1)
    disp = torch.matmul(normals, project_xyz)
    
    batch_size = pixels_xy.shape[0]
    K_ref_inv = intrinsics_inv[0, :3, :3]
    K_src = intrinsics[1:, :3, :3]
    num_src = K_src.shape[0]
    R_ref_inv = poses_pairs[0, :3, :3]
    R_src = poses_pairs[1:, :3, :3].permute(0, 2, 1)
    C_ref = poses_pairs[0, :3, 3]
    C_src = poses_pairs[1:, :3, 3]
    R_relative = torch.matmul(R_src, R_ref_inv)
    C_relative = C_ref[None, ...] - C_src
    
    tmp = torch.matmul(R_src, C_relative[..., None])
    tmp = torch.matmul(tmp[None, ...].expand(batch_size, num_src, 3, 1), normals.expand(batch_size, num_src, 3)[..., None].permute(0, 1, 3, 2))  
    tmp = R_relative[None, ...].expand(batch_size, num_src, 3, 3) + tmp / (disp[..., None] + 1e-10)
    tmp = torch.matmul(K_src[None, ...].expand(batch_size, num_src, 3, 3), tmp)
    Hom = torch.matmul(tmp, K_ref_inv[None, None, ...])
    
    patch_size = 5    # 3
    total_size = (patch_size * 2 + 1) ** 2
    offsets = torch.arange(-patch_size, patch_size + 1, device=device)
    offsets = torch.stack(torch.meshgrid(offsets, offsets)[::-1], dim=-1).view(1, -1, 2)  
    pixels_patch = pixels_xy.view(batch_size, 1, 2) + offsets.float() 
    ref_image = imgs_pairs[0, :, :]
    src_images = imgs_pairs[1:, :, :]
    
    def patch_homography(H, uv):      # 内部函数，对像素坐标进行单应变换
        N, Npx = uv.shape[:2]
        H = H.permute(1, 0, 2, 3)
        Nsrc = H.shape[0]
        H = H.view(Nsrc, N, -1, 3, 3)
        ones = torch.ones(uv.shape[:-1], device=uv.device).unsqueeze(-1)
        hom_uv = torch.cat((uv, ones), dim=-1)
        tmp = torch.einsum("vprik,pok->vproi", H, hom_uv)
        tmp = tmp.reshape(Nsrc, -1, 3)
        grid = tmp[..., :2] / (tmp[..., 2:] + 1e-8)
        return grid
    
    def compute_LNCC(ref_gray, src_grays):   # 内部函数，计算ncc
        ref_gray = ref_gray.permute(1, 0, 3, 2)  # [batch_size, 1, 1, 121]
        src_grays = src_grays.permute(1, 0, 3, 2)  # [batch_size, nsrc, 1, 121]

        ref_src = ref_gray * src_grays  # [batch_size, nsrc, 1, npatch]

        bs, nsrc, nc, npatch = src_grays.shape
        patch_size = int(sqrt(npatch))
        ref_gray = ref_gray.view(bs, 1, 1, patch_size, patch_size).view(-1, 1, patch_size, patch_size)
        src_grays = src_grays.view(bs, nsrc, 1, patch_size, patch_size).contiguous().view(-1, 1, patch_size, patch_size)
        ref_src = ref_src.view(bs, nsrc, 1, patch_size, patch_size).contiguous().view(-1, 1, patch_size, patch_size)

        ref_sq = ref_gray.pow(2)
        src_sq = src_grays.pow(2)

        filters = torch.ones(1, 1, patch_size, patch_size, device=ref_gray.device)
        padding = patch_size // 2

        ref_sum = F.conv2d(ref_gray, filters, stride=1, padding=padding)[:, :, padding, padding]
        src_sum = F.conv2d(src_grays, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)
        ref_sq_sum = F.conv2d(ref_sq, filters, stride=1, padding=padding)[:, :, padding, padding]
        src_sq_sum = F.conv2d(src_sq, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)
        ref_src_sum = F.conv2d(ref_src, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)

        u_ref = ref_sum / npatch
        u_src = src_sum / npatch

        cross = ref_src_sum - u_src * ref_sum - u_ref * src_sum + u_ref * u_src * npatch
        ref_var = ref_sq_sum - 2 * u_ref * ref_sum + u_ref * u_ref * npatch
        src_var = src_sq_sum - 2 * u_src * src_sum + u_src * u_src * npatch

        cc = cross * cross / (ref_var * src_var + 1e-5)  # [batch_size, nsrc, 1, npatch]
        ncc = 1 - cc
        ncc = torch.clamp(ncc, 0.0, 2.0)
        ncc, _ = torch.topk(ncc, 4, dim=1, largest=False)
        ncc = torch.mean(ncc, dim=1, keepdim=True)

        return ncc
    
    grid = patch_homography(Hom, pixels_patch)
    # 计算NCC
    grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
    grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
    sampled_gray_val = F.grid_sample(src_images.unsqueeze(1), grid.view(num_src, -1, 1, 2), align_corners=True)
    sampled_gray_val = sampled_gray_val.view(num_src, batch_size, total_size, 1)  # [nsrc, batch_size, 121, 1]
    pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
    pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
    grid = pixels_patch.detach()
    ref_gray_val = F.grid_sample(ref_image[None, None, ...], grid.view(1, -1, 1, 2), align_corners=True)
    ref_gray_val = ref_gray_val.view(1, batch_size, total_size, 1)
    ncc = compute_LNCC(ref_gray_val, sampled_gray_val).squeeze()
    ncc = ncc.detach()
    
    # 计算Harris角点响应得分
    ref_gray_val = ref_gray_val.reshape(batch_size, 1, int(sqrt(total_size)), int(sqrt(total_size)))      # 范围[0,1]
    kernel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).float().to(device)
    kernel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).float().to(device)
    grad_x = F.conv2d(ref_gray_val, kernel_x[None,...], stride=1, padding=1)
    grad_y = F.conv2d(ref_gray_val, kernel_y[None,...], stride=1, padding=1)
    grad_xx = grad_x * grad_x
    grad_yy = grad_y * grad_y
    grad_xy = grad_x * grad_y
    # 高斯平滑核
    sigma = 1        # others
    offsets = torch.arange(-patch_size, patch_size + 1, device=device)
    offsets = torch.stack(torch.meshgrid(offsets, offsets), dim=-1)
    kernel_mean = torch.exp(-(offsets[:,:,0]*offsets[:,:,0]+offsets[:,:,1]*offsets[:,:,1])/(2*sigma*sigma))    # 均值为0，标准差为sigma的高斯核
    kernel_mean = kernel_mean/kernel_mean.sum()
    kernel_mean = kernel_mean[None, None, :, :]
    # 均值平滑核
    # kernel_mean = torch.ones(1,1,patch_size*2+1,patch_size*2+1).float().to(self.device)/total_size   # 均值滤波
    grad_xx = F.conv2d(grad_xx, kernel_mean).reshape(batch_size)
    grad_yy = F.conv2d(grad_yy, kernel_mean).reshape(batch_size)
    grad_xy = F.conv2d(grad_xy, kernel_mean).reshape(batch_size)
    det_mat = grad_xx * grad_yy - grad_xy * grad_xy
    trace_mat = grad_xx + grad_yy
    scores = det_mat - 0.04 * trace_mat * trace_mat
    scores = torch.abs(scores)    # 加上绝对值
    stretch_factor = 0.2    
    scores = torch.pow(scores, stretch_factor)
    
    weight = scores * ncc
    
    # 可视化
    scores = scores.reshape(H, W)
    scores = scores.clip(0,1)
    ncc = ncc.reshape(H, W)
    ncc = ncc.clip(0,1)
    weight_t = weight.reshape(H, W)
    weight_t = weight_t.clip(0,1)
    
    img_ref = (ref_image.cpu().numpy()*255).astype(np.uint8)
    scores = (scores.cpu().numpy()*255).astype(np.uint8)
    ncc = (ncc.cpu().numpy()*255).astype(np.uint8)
    weight_t = (weight_t.cpu().numpy()*255).astype(np.uint8)
    
    # cv2.imwrite("img_ref.jpg", img_ref)
    # cv2.imwrite("harris.jpg", scores)
    # cv2.imwrite("ncc.jpg", ncc)
    # cv2.imwrite("weight.jpg", weight_t)
    
    cv2.imshow("img_ref", img_ref)
    cv2.imshow("harris", scores)
    cv2.imshow("ncc", ncc)
    cv2.imshow("weight", weight_t)
    cv2.waitKey(10000)

    return weight


if __name__ == "__main__":
    sample_data = torch.load("./example.pth")
    compute_colorloss_weight(sample_data)
    