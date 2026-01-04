import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

transform1 = transforms.CenterCrop((576, 768))
transform2 = transforms.CenterCrop((544, 736))

import numpy as np
import random

def get_random_crop_mask(image_size, crop_size, seed=42):
    h_orig, w_orig = image_size
    h_crop, w_crop = crop_size

    if h_crop > h_orig or w_crop > w_orig:
        raise ValueError("裁剪尺寸必须小于或等于原始图像尺寸。")
    if seed is not None:
        random.seed(seed)
        
    start_h = random.randint(0, h_orig - h_crop)
    start_w = random.randint(0, w_orig - w_crop)
    mask = np.zeros(image_size, dtype=bool)
    mask[start_h : start_h + h_crop, start_w : start_w + w_crop] = True

    return mask

def patchify(img, patch_size):
    img = img.unsqueeze(0)
    img = F.unfold(img, patch_size, stride=patch_size)
    img = img.transpose(2, 1).contiguous()
    return img.view(-1, patch_size, patch_size)

def depatchify(img, patch_size, output_size):
    img = img.view(1, -1, patch_size * patch_size).transpose(2, 1).contiguous()
    img = F.fold(img,
        output_size=output_size,
        kernel_size=patch_size,
        stride=patch_size
    )
    return img.squeeze(0)

def patched_depth_ranking_loss(surf_depth, mono_depth, patch_size=-1, margin=1e-4):
    if patch_size > 0:
        surf_depth_patches = patchify(surf_depth, patch_size).view(-1, patch_size * patch_size) # [N, P*P]
        mono_depth_patches = patchify(mono_depth, patch_size).view(-1, patch_size * patch_size)
    else:
        surf_depth_patches = surf_depth.reshape(-1).unsqueeze(0)
        mono_depth_patches = mono_depth.reshape(-1).unsqueeze(0)

    length = (surf_depth_patches.shape[1]) // 2 * 2
    rand_indices = torch.randperm(length)
    surf_depth_patches_rand = surf_depth_patches[:, rand_indices]
    mono_depth_patches_rand = mono_depth_patches[:, rand_indices]

    patch_rank_loss = torch.max(
        torch.sign(mono_depth_patches_rand[:, :length // 2] - mono_depth_patches_rand[:, length // 2:]) * \
            (surf_depth_patches_rand[:, length // 2:] - surf_depth_patches_rand[:, :length // 2]) + margin,
        torch.zeros_like(mono_depth_patches_rand[:, :length // 2], device=mono_depth_patches_rand.device)
    ).mean()

    return patch_rank_loss

def get_depth_ranking_loss(surf_depth, mono_depth, object_mask=None):
    depth_rank_loss = 0.0

    for transform in [transform1, transform2]:
        surf_depth_crop = transform(surf_depth)
        mono_depth_crop = transform(mono_depth.unsqueeze(0))

        object_mask_crop = None
        if object_mask is not None:
            object_mask_crop = transform(object_mask)
            surf_depth_crop[object_mask_crop.float() < 0.5] = -1e-4
            mono_depth_crop[object_mask_crop.float() < 0.5] = -1e-4

        depth_rank_loss += 0.5 * patched_depth_ranking_loss(surf_depth_crop, mono_depth_crop, patch_size=32)

    return depth_rank_loss

def get_depth_align(surf_depth, mono_depth, object_mask=None, patch_size=32):
    # Input: (N, H, W)
    _, H, W = surf_depth.shape

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    padded_H = H + pad_h
    padded_W = W + pad_w

    padded_surf_depth = F.pad(surf_depth, (0, pad_w, 0, pad_h))
    padded_mono_depth = F.pad(mono_depth, (0, pad_w, 0, pad_h))

    surf_depth_patches = patchify(padded_surf_depth, patch_size).view(-1, patch_size * patch_size) # [N, P*P]
    mono_depth_patches = patchify(padded_mono_depth, patch_size).view(-1, patch_size * patch_size)
    mono_depth_patches_ori = mono_depth_patches.clone()

    N = surf_depth_patches.shape[0]

    if object_mask is not None:
        object_mask = object_mask.float()
        padded_object_mask = F.pad(object_mask, (0, pad_w, 0, pad_h))
        object_mask_pathes = patchify(padded_object_mask, patch_size).view(-1, patch_size * patch_size)

        surf_depth_patches[object_mask_pathes.float() < 0.5] = 0
        mono_depth_patches[object_mask_pathes.float() < 0.5] = 0

        mono_depth_patches_ori[object_mask_pathes.float() > 0.5] = 0

    valid_mask = (surf_depth_patches != 0).sum(dim=1) > patch_size * patch_size * 0.5

    valid_surf_depth_patches = surf_depth_patches[valid_mask]
    valid_mono_depth_patches = mono_depth_patches[valid_mask]

    A = torch.stack([valid_mono_depth_patches, torch.ones_like(valid_mono_depth_patches)], dim=-1)
    B = valid_surf_depth_patches.unsqueeze(-1)

    A_T = A.transpose(-1, -2)
    A_T_A = torch.matmul(A_T, A)
    A_T_B = torch.matmul(A_T, B)

    s_t_params = torch.ones(N, 2, device=A.device) * torch.tensor([1.0, 0.0], device=A.device)
    try:
        results = torch.linalg.solve(A_T_A, A_T_B)
        s_t_params[valid_mask] = results.squeeze(-1)
    except torch.linalg.LinAlgError:
        # print("Warning: Failed to solve linear system. Returning default values.", A.shape, B.shape)
        return None
    
    s = s_t_params[:, 0].unsqueeze(-1)
    t = s_t_params[:, 1].unsqueeze(-1)

    transformed_depth_patches = mono_depth_patches_ori * s + t
    error_patches = torch.abs(transformed_depth_patches[valid_mask] - valid_surf_depth_patches)
    # error_mask = error_patches.mean(1) < 0.3
    # transformed_depth_patches[valid_mask][~error_mask] = 0.0
    transformed_depth_patches[~valid_mask] = 0.0
    transformed_depth_patches = transformed_depth_patches.view(-1, patch_size, patch_size)
    transformed_depth = depatchify(transformed_depth_patches, patch_size, (padded_H, padded_W))
    transformed_depth = transformed_depth[:, :H, :W]
    transformed_depth[transformed_depth < 0.0] = 0.0

    return transformed_depth

def get_depth_completion(surf_depth, mono_depth, object_mask=None, patch_size=32):
    # Input: (N, H, W)
    _, H, W = surf_depth.shape

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    padded_H = H + pad_h
    padded_W = W + pad_w

    padded_surf_depth = F.pad(surf_depth, (0, pad_w, 0, pad_h))
    padded_mono_depth = F.pad(mono_depth, (0, pad_w, 0, pad_h))

    surf_depth_patches = patchify(padded_surf_depth, patch_size).view(-1, patch_size * patch_size) # [N, P*P]
    mono_depth_patches = patchify(padded_mono_depth, patch_size).view(-1, patch_size * patch_size)
    mono_depth_patches_ori = mono_depth_patches.clone()

    N = surf_depth_patches.shape[0]

    if object_mask is not None:
        object_mask = object_mask.float()
        padded_object_mask = F.pad(object_mask, (0, pad_w, 0, pad_h))
        object_mask_pathes = patchify(padded_object_mask, patch_size).view(-1, patch_size * patch_size)

        surf_depth_patches[object_mask_pathes.float() < 0.5] = 0
        mono_depth_patches[object_mask_pathes.float() < 0.5] = 0

    valid_mask = (surf_depth_patches != 0).sum(dim=1) > patch_size * patch_size * 0.1

    valid_surf_depth_patches = surf_depth_patches[valid_mask]
    valid_mono_depth_patches = mono_depth_patches[valid_mask]

    A = torch.stack([valid_mono_depth_patches, torch.ones_like(valid_mono_depth_patches)], dim=-1)
    B = valid_surf_depth_patches.unsqueeze(-1)

    A_T = A.transpose(-1, -2)
    A_T_A = torch.matmul(A_T, A)
    A_T_B = torch.matmul(A_T, B)

    s_t_params = torch.ones(N, 2, device=A.device) * torch.tensor([1.0, 0.0], device=A.device)
    try:
        results = torch.linalg.solve(A_T_A, A_T_B)
        s_t_params[valid_mask] = results.squeeze(-1)
    except torch.linalg.LinAlgError:
        # print("Warning: Failed to solve linear system. Returning default values.", A.shape, B.shape)
        return None
    
    s = s_t_params[:, 0].unsqueeze(-1)
    t = s_t_params[:, 1].unsqueeze(-1)

    transformed_depth_patches = mono_depth_patches_ori * s + t
    transformed_depth_patches = transformed_depth_patches.view(-1, patch_size, patch_size)
    transformed_depth = depatchify(transformed_depth_patches, patch_size, (padded_H, padded_W))
    transformed_depth = transformed_depth[:, :H, :W]
    transformed_depth[transformed_depth < 0.0] = 0.0

    return transformed_depth

def get_random_depth_ranking_loss(surf_depth, mono_depth, object_mask=None):
    depth_rank_loss = 0.0
    img_size = surf_depth.shape[-2:]
    for crop_size in [(576, 768), (544, 736)]:
        crop_mask = get_random_crop_mask(img_size, crop_size)

        surf_depth_crop = surf_depth[:, crop_mask].view(-1, crop_size[0], crop_size[1])
        mono_depth_crop = mono_depth.unsqueeze(0)[:, crop_mask].view(-1, crop_size[0], crop_size[1])

        object_mask_crop = None
        if object_mask is not None:
            object_mask_crop = object_mask[:, crop_mask].view(-1, crop_size[0], crop_size[1])
            surf_depth_crop[object_mask_crop.float() < 0.5] = -1e-4
            mono_depth_crop[object_mask_crop.float() < 0.5] = -1e-4

        depth_rank_loss += 0.5 * patched_depth_ranking_loss(surf_depth_crop, mono_depth_crop, patch_size=32)

    return depth_rank_loss

def get_random_depth_boundary_loss(surf_depth, mono_depth, object_mask=None):
    depth_rank_loss = 0.0
    img_size = surf_depth.shape[-2:]
    for crop_size in [(640, 768), (768, 896)]:
        crop_mask = get_random_crop_mask(img_size, crop_size)

        surf_depth_crop = surf_depth[:, crop_mask].view(-1, crop_size[0], crop_size[1])
        mono_depth_crop = mono_depth.unsqueeze(0)[:, crop_mask].view(-1, crop_size[0], crop_size[1])

        object_mask_crop = None
        if object_mask is not None:
            object_mask_crop = object_mask[:, crop_mask].view(-1, crop_size[0], crop_size[1])
            surf_depth_crop[object_mask_crop.float() < 0.5] = -1e-4
            mono_depth_crop[object_mask_crop.float() < 0.5] = -1e-4

        patch_size = 64
        while patch_size >= 16:
            depth_rank_loss += 0.5 * patched_depth_boundary_loss(surf_depth_crop, mono_depth_crop, patch_size=patch_size)
            patch_size //= 2

    return depth_rank_loss

def patched_depth_boundary_loss(surf_depth, mono_depth, patch_size=-1, margin=1e-4):
    if patch_size > 0:
        surf_depth_patches = patchify(surf_depth, patch_size).view(-1, patch_size * patch_size) # [N, P*P]
        mono_depth_patches = patchify(mono_depth, patch_size).view(-1, patch_size * patch_size)
    else:
        surf_depth_patches = surf_depth.reshape(-1).unsqueeze(0)
        mono_depth_patches = mono_depth.reshape(-1).unsqueeze(0)

    mono_max_inv_depth, mono_max_indices = torch.max(mono_depth_patches, dim=1, keepdim=True)
    mono_min_inv_depth, mono_min_indices = torch.min(mono_depth_patches, dim=1, keepdim=True)

    surf_at_mono_max = torch.gather(surf_depth_patches, 1, mono_max_indices)
    surf_at_mono_min = torch.gather(surf_depth_patches, 1, mono_min_indices)

    surf_max_inv_depth, _ = torch.max(surf_depth_patches, dim=1, keepdim=True)
    surf_min_inv_depth, _ = torch.min(surf_depth_patches, dim=1, keepdim=True)

    boundary_rank_loss = (surf_max_inv_depth - surf_at_mono_max).mean() + (surf_at_mono_min - surf_min_inv_depth).mean()

    total_loss = boundary_rank_loss

    return total_loss

def get_random_depth_contrastives_loss(surf_depth, mono_depth, object_mask=None):
    depth_rank_loss = 0.0
    img_size = surf_depth.shape[-2:]
    for crop_size in [(640, 768), (768, 896)]:
        crop_mask = get_random_crop_mask(img_size, crop_size)

        surf_depth_crop = surf_depth[:, crop_mask].view(-1, crop_size[0], crop_size[1])
        mono_depth_crop = mono_depth.unsqueeze(0)[:, crop_mask].view(-1, crop_size[0], crop_size[1])

        object_mask_crop = None
        if object_mask is not None:
            object_mask_crop = object_mask[:, crop_mask].view(-1, crop_size[0], crop_size[1])
            surf_depth_crop[object_mask_crop.float() < 0.5] = -1e-4
            mono_depth_crop[object_mask_crop.float() < 0.5] = -1e-4

        depth_rank_loss += 0.5 * patched_depth_contrastives_loss(surf_depth_crop, mono_depth_crop, patch_size=64)

    return depth_rank_loss

def patched_depth_contrastives_loss(surf_depth, mono_depth, patch_size=-1, margin=1e-4):
    if patch_size > 0:
        surf_depth_patches = patchify(surf_depth, patch_size).view(-1, patch_size * patch_size) # [N, P*P]
        mono_depth_patches = patchify(mono_depth, patch_size).view(-1, patch_size * patch_size)
    else:
        surf_depth_patches = surf_depth.reshape(-1).unsqueeze(0)
        mono_depth_patches = mono_depth.reshape(-1).unsqueeze(0)

    mono_max_inv_depth, mono_max_indices = torch.max(mono_depth_patches, dim=1, keepdim=True)
    mono_min_inv_depth, mono_min_indices = torch.min(mono_depth_patches, dim=1, keepdim=True)

    surf_at_mono_max = torch.gather(surf_depth_patches, 1, mono_max_indices)
    surf_at_mono_min = torch.gather(surf_depth_patches, 1, mono_min_indices)

    surf_max_inv_depth, _ = torch.max(surf_depth_patches, dim=1, keepdim=True)
    surf_min_inv_depth, _ = torch.min(surf_depth_patches, dim=1, keepdim=True)

    boundary_rank_loss = (surf_max_inv_depth - surf_at_mono_max).mean() + (surf_at_mono_min - surf_min_inv_depth).mean()

    # random sample
    # num_samples = int(surf_depth_patches.shape[1] / 4)
    # rand_indices = torch.randperm(surf_depth_patches.shape[1])
    # sampled_indices = rand_indices[:num_samples]

    # surf_depth_patches_rand = surf_depth_patches[:, sampled_indices]
    # mono_depth_patches_rand = mono_depth_patches[:, sampled_indices]

    # mono_dist_near = mono_max_inv_depth - mono_depth_patches_rand
    # mono_dist_far = mono_depth_patches_rand - mono_min_inv_depth
    # surf_dist_near = surf_at_mono_max - surf_depth_patches_rand
    # surf_dist_far = surf_depth_patches_rand - surf_at_mono_min

    mono_dist_near = mono_max_inv_depth - mono_depth_patches
    mono_dist_far = mono_depth_patches - mono_min_inv_depth
    surf_dist_near = surf_at_mono_max - surf_depth_patches
    surf_dist_far = surf_depth_patches - surf_at_mono_min

    sign_of_closeness = torch.sign(mono_dist_near - mono_dist_far)
    predicted_closeness_diff = surf_dist_far - surf_dist_near

    contrastive_loss = torch.max(
        sign_of_closeness * predicted_closeness_diff + margin,
        torch.zeros_like(predicted_closeness_diff)
    ).mean()

    total_loss = 0.5 * boundary_rank_loss + 0.05 * contrastive_loss

    return total_loss