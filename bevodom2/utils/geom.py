"""
Geometry transformation utilities.

Provides coordinate frame transforms, intrinsics manipulation,
grid generation, point-cloud projection, and related operations.
"""

import torch
import numpy as np

EPS = 1e-6


# ============================================================
#  Debug helpers
# ============================================================

def print_(name, tensor):
    """Print tensor name, values, and shape."""
    tensor = tensor.detach().cpu().numpy()
    print(name, tensor, tensor.shape)


def print_stats(name, tensor):
    """Print min / mean / max statistics of a tensor."""
    shape = tensor.shape
    tensor = tensor.detach().cpu().numpy()
    print('%s (%s) min = %.2f, mean = %.2f, max = %.2f' % (
        name, tensor.dtype, np.min(tensor), np.mean(tensor), np.max(tensor)), shape)


def strnum(x):
    """Format a number as a compact string; drop leading zero when < 1."""
    s = '%g' % x
    if '.' in s:
        if x < 1.0:
            s = s[s.index('.'):]
    return s


# ============================================================
#  Matrix multiplication & sequence-dim packing
# ============================================================

def matmul2(mat1, mat2):
    """Simple wrapper around torch.matmul for two matrices."""
    return torch.matmul(mat1, mat2)


def pack_seqdim(tensor, B):
    """Merge (B, S, ...) tensor into (B*S, ...)."""
    shapelist = list(tensor.shape)
    B_, S = shapelist[:2]
    assert B == B_
    otherdims = shapelist[2:]
    tensor = torch.reshape(tensor, [B * S] + otherdims)
    return tensor


def unpack_seqdim(tensor, B):
    """Restore (B*S, ...) tensor back to (B, S, ...)."""
    shapelist = list(tensor.shape)
    BS = shapelist[0]
    assert BS % B == 0
    otherdims = shapelist[1:]
    S = int(BS / B)
    tensor = torch.reshape(tensor, [B, S] + otherdims)
    return tensor


# ============================================================
#  Normalization
# ============================================================

def normalize_single(d):
    """Min-max normalize a single tensor to [0, 1]."""
    dmin = torch.min(d)
    dmax = torch.max(d)
    d = (d - dmin) / (EPS + (dmax - dmin))
    return d


def normalize(d):
    """Per-batch-element min-max normalization for (B, ...) tensors."""
    out = torch.zeros(d.size())
    if d.is_cuda:
        out = out.cuda()
    B = list(d.size())[0]
    for b in range(B):
        out[b] = normalize_single(d[b])
    return out


def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    """Compute mean over masked region (x and mask must share the same shape)."""
    for (a, b) in zip(x.size(), mask.size()):
        assert a == b
    prod = x * mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS + torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS + torch.sum(mask, dim=dim, keepdim=keepdim)
    mean = numer / denom
    return mean


# ============================================================
#  Grid / point-cloud generation
# ============================================================

def meshgrid2d(B, Y, X, stack=False, norm=False, device='cuda'):
    """Create a B x Y x X 2-D coordinate grid."""
    grid_y = torch.linspace(0.0, Y - 1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2d(grid_y, grid_x, Y, X)

    if stack:
        # Stack in xy order (matches F.grid_sample convention)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x


def meshgrid3d(B, Z, Y, X, stack=False, norm=False, device='cuda'):
    """Create a B x Z x Y x X 3-D coordinate grid."""
    grid_z = torch.linspace(0.0, Z - 1, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, Y - 1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)

    if norm:
        grid_z, grid_y, grid_x = normalize_grid3d(
            grid_z, grid_y, grid_x, Z, Y, X)

    if stack:
        # Stack in xyz order (matches F.grid_sample convention)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x


def gridcloud3d(B, Z, Y, X, norm=False, device='cuda'):
    """Flatten a 3-D grid into a B x N x 3 point cloud."""
    grid_z, grid_y, grid_x = meshgrid3d(B, Z, Y, X, norm=norm, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])
    xyz = torch.stack([x, y, z], dim=2)  # B x N x 3
    return xyz


def normalize_grid2d(grid_y, grid_x, Y, X, clamp_extreme=True):
    """Normalize 2-D grid coordinates to [-1, 1]."""
    grid_y = 2.0 * (grid_y / float(Y - 1)) - 1.0
    grid_x = 2.0 * (grid_x / float(X - 1)) - 1.0
    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
    return grid_y, grid_x


def normalize_grid3d(grid_z, grid_y, grid_x, Z, Y, X, clamp_extreme=True):
    """Normalize 3-D grid coordinates to [-1, 1]."""
    grid_z = 2.0 * (grid_z / float(Z - 1)) - 1.0
    grid_y = 2.0 * (grid_y / float(Y - 1)) - 1.0
    grid_x = 2.0 * (grid_x / float(X - 1)) - 1.0
    if clamp_extreme:
        grid_z = torch.clamp(grid_z, min=-2.0, max=2.0)
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
    return grid_z, grid_y, grid_x


# ============================================================
#  4x4 rigid-body transform (RT) operations
# ============================================================

def eye_4x4(B, device='cuda'):
    """Create B identity 4x4 matrices."""
    rt = torch.eye(4, device=torch.device(device)).view(1, 4, 4).repeat([B, 1, 1])
    return rt


def safe_inverse(a):
    """Batch-invert 4x4 rigid-body transforms (uses R^T for rotation)."""
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1, 2)
    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])
    return inv


def safe_inverse_single(a):
    """Invert a single 4x4 rigid-body transform."""
    r, t = split_rt_single(a)
    t = t.view(3, 1)
    r_transpose = r.t()
    inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)
    bottom_row = a[3:4, :]  # [0, 0, 0, 1]
    inv = torch.cat([inv, bottom_row], 0)
    return inv


def apply_4x4(RT, xyz):
    """Apply a 4x4 transform RT to point cloud xyz (B x N x 3)."""
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:, :, 0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)  # B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    xyz2 = xyz2[:, :, :3]
    return xyz2


def apply_r4x4(xyz, RT):
    """Right-multiply: xyz @ RT, return first 3 columns."""
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:, :, 0:1])
    xyz1 = torch.cat([xyz, ones], 2)  # B x N x 4
    xyz2 = torch.matmul(xyz1, RT)
    xyz2 = xyz2[:, :, :3]
    return xyz2


def get_camM_T_camXs(origin_T_camXs, ind=0):
    """Compute transforms from camera frame `ind` to all other frames."""
    B, S = list(origin_T_camXs.shape)[0:2]
    camM_T_camXs = torch.zeros_like(origin_T_camXs)
    for b in range(B):
        camM_T_origin = safe_inverse_single(origin_T_camXs[b, ind])
        for s in range(S):
            camM_T_camXs[b, s] = torch.matmul(camM_T_origin, origin_T_camXs[b, s])
    return camM_T_camXs


# ============================================================
#  Rotation + translation split / merge
# ============================================================

def split_rt_single(rt):
    """Extract R (3x3) and t (3,) from a single 4x4 matrix."""
    r = rt[:3, :3]
    t = rt[:3, 3].view(3)
    return r, t


def split_rt(rt):
    """Extract R (Bx3x3) and t (Bx3) from a batch of 4x4 matrices."""
    r = rt[:, :3, :3]
    t = rt[:, :3, 3].view(-1, 3)
    return r, t


def merge_rt(r, t):
    """Merge R (Bx3x3) and t (Bx3) into a batch of 4x4 transforms."""
    B, C, D = list(r.shape)
    B2, D2 = list(t.shape)
    assert C == 3 and D == 3
    assert B == B2 and D2 == 3
    t = t.view(B, 3)
    rt = eye_4x4(B, device=t.device)
    rt[:, :3, :3] = r
    rt[:, :3, 3] = t
    return rt


# ============================================================
#  Camera intrinsics operations
# ============================================================

def scale_intrinsics(K, sx, sy):
    """Scale camera intrinsics by (sx, sy)."""
    fx, fy, x0, y0 = split_intrinsics(K)
    fx = fx * sx
    fy = fy * sy
    x0 = x0 * sx
    y0 = y0 * sy
    K = merge_intrinsics(fx, fy, x0, y0)
    return K


def split_intrinsics(K):
    """Extract fx, fy, x0, y0 from a Bx3x3 or Bx4x4 intrinsics matrix."""
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    x0 = K[:, 0, 2]
    y0 = K[:, 1, 2]
    return fx, fy, x0, y0


def merge_intrinsics(fx, fy, x0, y0):
    """Assemble fx, fy, x0, y0 into a Bx4x4 intrinsics matrix."""
    B = list(fx.shape)[0]
    K = torch.zeros(B, 4, 4, dtype=torch.float32, device=fx.device)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = x0
    K[:, 1, 2] = y0
    K[:, 2, 2] = 1.0
    K[:, 3, 3] = 1.0
    return K


# ============================================================
#  LRT (length + RT) list operations
# ============================================================

def merge_rtlist(rlist, tlist):
    """Merge rotation list (B,N,3,3) and translation list (B,N,3) into RT list."""
    B, N, D, E = list(rlist.shape)
    assert D == 3 and E == 3
    B, N, F = list(tlist.shape)
    assert F == 3

    __p = lambda x: pack_seqdim(x, B)
    __u = lambda x: unpack_seqdim(x, B)
    rlist_, tlist_ = __p(rlist), __p(tlist)
    rtlist_ = merge_rt(rlist_, tlist_)
    rtlist = __u(rtlist_)
    return rtlist


def split_lrtlist(lrtlist):
    """Split BxNx19 tensor into lenlist (BxNx3) and rtlist (BxNx4x4)."""
    B, N, D = list(lrtlist.shape)
    assert D == 19
    lenlist = lrtlist[:, :, :3].reshape(B, N, 3)
    ref_T_objs_list = lrtlist[:, :, 3:].reshape(B, N, 4, 4)
    return lenlist, ref_T_objs_list


def merge_lrtlist(lenlist, rtlist):
    """Merge lenlist (BxNx3) and rtlist (BxNx4x4) into BxNx19 tensor."""
    B, N, D = list(lenlist.shape)
    assert D == 3
    B2, N2, E, F = list(rtlist.shape)
    assert B == B2 and N == N2
    assert E == 4 and F == 4
    rtlist = rtlist.reshape(B, N, 16)
    lrtlist = torch.cat([lenlist, rtlist], axis=2)
    return lrtlist


def apply_4x4_to_lrtlist(Y_T_X, lrtlist_X):
    """Left-multiply each RT in lrtlist by Y_T_X."""
    B, N, D = list(lrtlist_X.shape)
    assert D == 19
    B2, E, F = list(Y_T_X.shape)
    assert B2 == B
    assert E == 4 and F == 4

    lenlist, rtlist_X = split_lrtlist(lrtlist_X)
    Y_T_Xs = Y_T_X.unsqueeze(1).repeat(1, N, 1, 1)
    Y_T_Xs_ = Y_T_Xs.view(B * N, 4, 4)
    rtlist_X_ = rtlist_X.reshape(B * N, 4, 4)
    rtlist_Y_ = matmul2(Y_T_Xs_, rtlist_X_)
    rtlist_Y = rtlist_Y_.reshape(B, N, 4, 4)
    lrtlist_Y = merge_lrtlist(lenlist, rtlist_Y)
    return lrtlist_Y


def apply_4x4_to_lrt(Y_T_X, lrt_X):
    """Apply a 4x4 transform to a single lrt (Bx19)."""
    B, D = list(lrt_X.shape)
    assert D == 19
    B2, E, F = list(Y_T_X.shape)
    assert B2 == B
    assert E == 4 and F == 4
    return apply_4x4_to_lrtlist(Y_T_X, lrt_X.unsqueeze(1)).squeeze(1)


def get_xyzlist_from_lenlist(lenlist):
    """Generate 8 corner points (BxNx8x3) from a size list (BxNx3)."""
    B, N, D = list(lenlist.shape)
    assert D == 3
    lx, ly, lz = torch.unbind(lenlist, axis=2)

    xs = torch.stack([lx/2., lx/2., -lx/2., -lx/2.,
                      lx/2., lx/2., -lx/2., -lx/2.], axis=2)
    ys = torch.stack([ly/2., ly/2., ly/2., ly/2.,
                      -ly/2., -ly/2., -ly/2., -ly/2.], axis=2)
    zs = torch.stack([lz/2., -lz/2., -lz/2., lz/2.,
                      lz/2., -lz/2., -lz/2., lz/2.], axis=2)

    xyzlist = torch.stack([xs, ys, zs], axis=3)  # B x N x 8 x 3
    return xyzlist


def get_xyzlist_from_lrtlist(lrtlist, include_clist=False):
    """Get world-frame 8-corner coordinates from lrtlist (BxNx19)."""
    B, N, D = list(lrtlist.shape)
    assert D == 19

    lenlist, rtlist = split_lrtlist(lrtlist)
    xyzlist_obj = get_xyzlist_from_lenlist(lenlist)  # B x N x 8 x 3

    rtlist_ = rtlist.reshape(B * N, 4, 4)
    xyzlist_obj_ = xyzlist_obj.reshape(B * N, 8, 3)
    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_obj_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 8, 3)

    if include_clist:
        clist_cam = get_clist_from_lrtlist(lrtlist).unsqueeze(2)
        xyzlist_cam = torch.cat([xyzlist_cam, clist_cam], dim=2)
    return xyzlist_cam


def get_clist_from_lrtlist(lrtlist):
    """Extract object center coordinates (BxNx3) from lrtlist."""
    B, N, D = list(lrtlist.shape)
    assert D == 19

    lenlist, rtlist = split_lrtlist(lrtlist)
    xyzlist_obj = torch.zeros((B, N, 1, 3), device=lrtlist.device)

    rtlist_ = rtlist.reshape(B * N, 4, 4)
    xyzlist_obj_ = xyzlist_obj.reshape(B * N, 1, 3)
    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_obj_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 3)
    return xyzlist_cam


# ============================================================
#  Pixel <-> camera coordinate projection
# ============================================================

def wrap2pi(rad_angle):
    """Wrap angle to [-pi, pi]."""
    return torch.atan2(torch.sin(rad_angle), torch.cos(rad_angle))


def xyd2pointcloud(xyd, pix_T_cam):
    """Back-project pixel coords + depth (BxNx3) to a 3-D point cloud."""
    B, N, C = list(xyd.shape)
    assert C == 3
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = pixels2camera(xyd[:, :, 0], xyd[:, :, 1], xyd[:, :, 2],
                        fx, fy, x0, y0)
    return xyz


def pixels2camera(x, y, z, fx, fy, x0, y0):
    """Back-project pixel (x, y) + depth z to camera frame; returns BxNx3."""
    B = x.shape[0]

    fx = torch.reshape(fx, [B, 1])
    fy = torch.reshape(fy, [B, 1])
    x0 = torch.reshape(x0, [B, 1])
    y0 = torch.reshape(y0, [B, 1])

    x = torch.reshape(x, [B, -1])
    y = torch.reshape(y, [B, -1])
    z = torch.reshape(z, [B, -1])

    # Un-project
    x = (z / fx) * (x - x0)
    y = (z / fy) * (y - y0)

    xyz = torch.stack([x, y, z], dim=2)  # B x N x 3
    return xyz


def camera2pixels(xyz, pix_T_cam):
    """Project camera coordinates (BxNx3) to pixel coordinates (BxNx2)."""
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    x, y, z = torch.unbind(xyz, dim=-1)
    B = list(z.shape)[0]

    fx = torch.reshape(fx, [B, 1])
    fy = torch.reshape(fy, [B, 1])
    x0 = torch.reshape(x0, [B, 1])
    y0 = torch.reshape(y0, [B, 1])
    x = torch.reshape(x, [B, -1])
    y = torch.reshape(y, [B, -1])
    z = torch.reshape(z, [B, -1])

    eps = 1e-4
    z = torch.clamp(z, min=eps)
    x = (x * fx) / z + x0
    y = (y * fy) / z + y0
    xy = torch.stack([x, y], dim=-1)
    return xy
