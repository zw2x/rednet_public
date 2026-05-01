"""Data structures and utilities for rigid objects"""

from __future__ import annotations
from typing import Optional
import dataclasses

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

from .ops import to_pairwise_mask, to_one_hot, mask_mean
from .common_utils import get_logger

log = get_logger(__name__)


@dataclasses.dataclass(kw_only=True)
class Rigid:
    # shape, [bsz, num, 3, 3], matrix representation. columns correspond to unit x, y, z axis
    rotation: torch.FloatTensor
    center: torch.FloatTensor = None  # shape, [bsz, num, 3]
    mask: torch.BoolTensor = None  # shape, [bsz, num], mask for the rigid object. 1 for valid, 0 for invalid
    # shape, [bsz, num, 4], quaternion. we do not store axis-angle representations, becuase it is simple to convert to/from quaternion
    quat: Optional[torch.FloatTensor] = None

    def __post_init__(self):
        self.rotation = self.rotation.to(dtype=torch.float32)

        if self.mask is None:
            self.mask = torch.ones(self.shape, device=self.device, dtype=torch.bool)
        self.mask = self.mask.to(dtype=torch.bool)

        if self.center is None:
            self.center = torch.zeros(self.shape + (3,), device=self.device, dtype=torch.float32)
        self.center = self.center.to(dtype=torch.float32)

        assert self.rotation.shape == self.shape + (3, 3), f"rotation shape {self.rotation.shape} is not valid"
        assert self.center.shape == self.shape + (3,), f"center shape {self.center.shape} is not valid"
        assert self.mask.shape == self.shape, f"mask shape {self.mask.shape} is not valid"

    def to(self, *args, **kwargs) -> Rigid:
        self.rotation = self.rotation.to(*args, **kwargs)
        self.center = self.center.to(*args, **kwargs)
        self.mask = self.mask.to(*args, **kwargs)
        if self.quat is not None:
            self.quat = self.quat.to(*args, **kwargs)
        return self

    @property
    def shape(self):
        return self.center.shape[:-1]

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def device(self):
        return self.center.device

    @staticmethod
    def from_points(
        origin: torch.Tensor,
        x_axis: torch.Tensor,
        xy_plane: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        to_rigid: bool = True,
        eps: float = 1e-6,
    ) -> Rigid | torch.Tensor:
        x_axis = normalize(x_axis - origin, eps=eps)
        z_axis = normalize(torch.cross(x_axis, xy_plane - origin, dim=-1), eps=eps)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        rot = torch.stack([x_axis, y_axis, z_axis], dim=-1)

        if mask is not None:
            mask = mask.to(dtype=torch.bool)
            rot = rot * mask[..., None, None] + ~mask[..., None, None] * torch.eye(3, device=rot.device)
            origin = origin.masked_fill(~mask[..., None], 0.0)

        if to_rigid:
            return Rigid(rotation=rot, center=origin, mask=mask)
        else:
            return rot

    def apply_to_point(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Rigid transformation
        """
        assert pos.ndim == self.ndim + 1
        # apply rotation and translation
        new_pos = self.rotation @ pos[..., None] + self.center[..., None]
        return new_pos.squeeze(-1)

    def inverse_apply_to_point(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Inverse rigid transformation. Used to align positions to local referrence frames.
        """
        assert pos.ndim == self.ndim + 1
        # apply rotation and translation
        new_pos = self.rotation.transpose(-1, -2) @ (pos - self.center)[..., None]
        return new_pos.squeeze(-1)

    def __getitem__(self, index) -> Rigid:
        if not isinstance(index, tuple):
            index = (index,)
        new_rotation = self.rotation[index + (slice(None), slice(None))]
        new_center = self.center[index + (slice(None),)]
        new_mask = self.mask[index]
        return Rigid(rotation=new_rotation, center=new_center, mask=new_mask)

    def apply(self, other: Rigid) -> Rigid:
        assert self.shape == other.shape
        new_rotation = self.rotation @ other.rotation
        new_center = self.rotation @ other.center + self.center
        new_mask = self.mask & other.mask
        return Rigid(rotation=new_rotation, center=new_center, mask=new_mask)

    def inverse(self) -> Rigid:
        new_rotation = self.rotation.transpose(-1, -2)
        new_center = -self.rotation.transpose(-1, -2) @ self.center
        new_mask = self.mask
        return Rigid(rotation=new_rotation, center=new_center, mask=new_mask)

    # conversion
    def to_axis_angle(self, only_rotation_angle: bool = False, eps: float = 1e-6) -> torch.FloatTensor:
        """
        Rodrigues formula: R = I + sin(theta) * S + (1 - cos(theta)) * S^2
        unit vector axis = [x, y, z]
        S = [
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ]
        R - R^T = 2 * sin(theta) * S
        if theta in (0, pi), then [x, y, z] = normalize([r21 - r12, r02 - r20, r10 - r01])
        """
        # radian (0, \pi). clamp to avoid numerical errors during grad
        theta = torch.arccos(((trace(self.rotation) - 1) / 2).clamp(min=-1 + eps, max=1 - eps))
        if only_rotation_angle:
            return theta
        # axis = normalize([r21 - r12, r02 - r20, r10 - r01]). unit vector
        axis = self.rotation.new_zeros(theta.shape + (3,))
        axis[..., 0] = self.rotation[..., 2, 1] - self.rotation[..., 1, 2]
        axis[..., 1] = self.rotation[..., 0, 2] - self.rotation[..., 2, 0]
        axis[..., 2] = self.rotation[..., 1, 0] - self.rotation[..., 0, 1]
        axis = normalize(axis)
        return torch.cat([theta.unsqueeze(-1), axis], dim=-1)

    def to_quat(self, eps: float = 1e-6) -> torch.FloatTensor:
        # quaternion: [w, x, y, z]
        # w = cos(theta/2)
        w = torch.sqrt((1 + torch.trace(self.rotation)).clamp(min=eps)) / 2
        x = self.rotation[..., 2, 1] - self.rotation[..., 1, 2]
        y = self.rotation[..., 0, 2] - self.rotation[..., 2, 0]
        z = self.rotation[..., 1, 0] - self.rotation[..., 0, 1]
        v = normalize(torch.stack([x, y, z], dim=-1)) * (1 - w**2)
        return torch.stack([w.unsqueeze(-1), v], dim=-1)

    @classmethod
    def from_quat(cls, quat, center=None) -> Rigid:
        w, x, y, z = quat.unbind(-1)
        rot = [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
        rot = torch.stack([torch.stack(row, dim=-1) for row in rot], dim=-2)
        return cls(rotation=rot, center=center)

    @classmethod
    def from_graham_schmidt(
        cls, neg_x_axis: torch.Tensor, origin: torch.Tensor, xy_plane: torch.Tensor, eps: float = 1e-10
    ) -> Rigid:
        return from_graham_schmidt(neg_x_axis, origin, xy_plane, eps=eps)

    # alignment
    @staticmethod
    def kabsch(qry_pos, tgt_pos, *, pos_mask=None, weights=None):
        return weighted_align(qry_pos, tgt_pos, weights=weights, mask=pos_mask, return_transforms=False)

    def fape(self, query_rigid: Rigid, mask=None, weights=None):
        return

    def pae(self, query_rigid: Rigid, mask=None, weights=None, chain_mask=None):
        return

    def lddt(self, pos, pos_mask=None, weights=None, cutoffs=None):
        return

    # sampling

    @classmethod
    def identity(cls, shape, device=None) -> Rigid:
        rotation = torch.eye(3, device=device).expand(shape + (3, 3))
        center = torch.zeros(shape + (3,), device=device)
        mask = torch.ones(shape, device=device, dtype=torch.bool)
        return cls(rotation=rotation, center=center, mask=mask)

    @classmethod
    def uniform_quat(cls, shape, device=None) -> torch.FloatTensor:
        # Shoemake's method. ref https://lavalle.pl/planning/node198.html
        if isinstance(shape, int):
            shape = (shape,)
        u = torch.rand(shape + (3,), device=device)
        r1, r2 = torch.sqrt(1 - u[..., 0]), torch.sqrt(u[..., 0])
        theta_1 = 2 * torch.pi * u[..., 1]
        theta_2 = 2 * torch.pi * u[..., 2]
        w = torch.sin(theta_1) * r1
        x = torch.cos(theta_1) * r1
        y = torch.sin(theta_2) * r2
        z = torch.cos(theta_2) * r2
        quat = torch.stack([w, x, y, z], dim=-1)
        # quat = normalize(quat, dim=-1)
        return quat

    @classmethod
    def uniform(cls, shape, device=None, zero_center=False) -> Rigid:
        if isinstance(shape, int):
            shape = (shape,)
        quat = cls.uniform_quat(shape, device=device)
        # uniform sampling positions
        center = (
            torch.rand(shape + (3,), device=device) if not zero_center else torch.zeros(shape + (3,), device=device)
        )
        rigid = cls.from_quat(quat, center)
        return rigid

    # interpolation
    @staticmethod
    def slerp(p, q, t):
        raise NotImplementedError("SLERP is not implemented for Rigid objects")


def normalize(v, dim=-1, eps: float = 1e-6):
    """Normalize a vector"""
    norm = torch.norm(v, dim=dim, keepdim=True).clamp(min=eps)
    return v / norm


def trace(mat: torch.Tensor, dim1=-2, dim2=-1) -> torch.Tensor:
    """Trace of a matrix"""
    return torch.sum(torch.diagonal(mat, dim1=dim1, dim2=dim2), dim=-1)


def dotprod(a: torch.Tensor, b: torch.Tensor, dim=-1):
    """Dot product of two tensors"""
    return torch.sum(a * b, dim=dim)


def compute_distances(
    pos: torch.Tensor, mask=None, pos_y=None, mask_y=None, mask_fillin: float = 0.0, squared=False, eps=1e-6
) -> torch.Tensor:
    mask_y = mask if mask_y is None else mask_y
    pos_y = pos if pos_y is None else pos_y

    d = torch.sum((pos_y[..., None, :, :] - pos[..., :, None, :]) ** 2, dim=-1)
    if not squared:
        d = (d + eps).sqrt()
    if mask is not None:
        d = d.masked_fill(~mask.bool()[..., :, None], mask_fillin)
        # pair_mask = mask_y[..., None, :] * mask[..., :, None]
    if mask_y is not None:
        d = d.masked_fill(~mask_y.bool()[..., None, :], mask_fillin)
    return d


def from_graham_schmidt(neg_x_axis: torch.Tensor, origin: torch.Tensor, xy_plane: torch.Tensor, eps: float = 1e-10):
    x_axis = origin - neg_x_axis
    xy_plane = xy_plane - origin
    return Rigid(rotation=_graham_schmidt(x_axis, xy_plane, eps), center=origin)


def _graham_schmidt(x_axis: torch.Tensor, xy_plane: torch.Tensor, eps: float = 1e-10):
    e1 = xy_plane

    denom = torch.sqrt((x_axis**2).sum(dim=-1, keepdim=True) + eps)
    x_axis = x_axis / denom
    dot = (x_axis * e1).sum(dim=-1, keepdim=True)
    e1 = e1 - x_axis * dot
    denom = torch.sqrt((e1**2).sum(dim=-1, keepdim=True) + eps)
    e1 = e1 / denom
    e2 = torch.cross(x_axis, e1, dim=-1)

    rots = torch.stack([x_axis, e1, e2], dim=-1)

    return rots


# adopted from alphafold3-pytorch
@torch.autocast(device_type="cuda", enabled=False)
def weighted_align(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    weights: None | torch.Tensor = None,
    mask: None | torch.Tensor = None,
    return_transforms: bool = False,
):
    """Compute the weighted rigid alignment.

    The check for ambiguous rotation and low rank of cross-correlation between aligned point
    clouds is inspired by
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_alignment.html.

    :param pred_coords: Predicted coordinates.
    :param true_coords: True coordinates.
    :param weights: Weights for each atom.
    :param mask: The mask for variable lengths.
    :param return_transform: Whether to return the transformation matrix.
    :return: The optimally aligned coordinates.

    Args:
        qry_points: [bsz, num_points, 3]
    """
    pred_coords = pred_coords.to(dtype=torch.float32)
    true_coords = true_coords.to(dtype=torch.float32)
    batch_size, num_points, dim = pred_coords.shape

    if weights is None:
        # if no weights are provided, assume uniform weights
        weights = torch.ones_like(pred_coords[..., 0])

    if mask is not None:
        # zero out all predicted and true coordinates where not an atom
        pred_coords = pred_coords.masked_fill(~mask[..., None], 0.0)
        true_coords = true_coords.masked_fill(~mask[..., None], 0.0)
        weights = weights.masked_fill(~mask, 0.0)

    # Take care of weights broadcasting for coordinate dimension
    weights = rearrange(weights, "b n -> b n 1")

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        log.warning(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix
    cov_matrix = einsum(weights * true_coords_centered, pred_coords_centered, "b n i, b n j -> b i j")

    # Compute the SVD of the covariance matrix
    U, S, V = torch.svd(cov_matrix)
    U_T = U.transpose(-2, -1)

    # Catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        log.warning(
            "Warning: Excessively low rank of cross-correlation between aligned point clouds. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    det = torch.det(einsum(V, U_T, "b i j, b j k -> b i k"))

    # Ensure proper rotation matrix with determinant 1
    diag = torch.eye(dim, dtype=det.dtype, device=det.device)
    diag = repeat(diag, "i j -> b i j", b=batch_size).clone()

    diag[:, -1, -1] = det
    rot_matrix = einsum(V, diag, U_T, "b i j, b j k, b k l -> b i l")

    # Apply the rotation and translation
    true_aligned_coords = einsum(rot_matrix, true_coords_centered, "b i j, b n j -> b n i") + pred_centroid
    true_aligned_coords.detach_()

    if return_transforms:
        translation = true_centroid - einsum(rot_matrix, pred_centroid, "b i j, b ... j -> b ... i")
        return true_aligned_coords, rot_matrix, translation

    return true_aligned_coords


def compute_smooth_lddt(
    pred_coords,
    true_coords,
    *,
    atom_mask: torch.Tensor,
    lddt_thresholds=torch.tensor([0.5, 1.0, 2.0, 4.0]),
    cutoff=15.0,
    return_dists=False,
    pred_dists=None,
    true_dists=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    pred_coords: predicted coordinates
    true_coords: true coordinates
    """
    # Compute distances between all pairs of atoms
    if pred_dists is None:
        assert pred_coords.ndim == 3, "pred_coords should be of shape [bsz, num_atoms, 3]"
        pred_dists = compute_distances(pred_coords, mask=atom_mask)
    if true_dists is None:
        assert true_coords.ndim == 3, "true_coords should be of shape [bsz, num_atoms, 3]"
        true_dists = compute_distances(true_coords, mask=atom_mask)

    # Compute distance difference for all pairs of atoms
    dist_diff = torch.abs(true_dists - pred_dists)

    # Compute epsilon values
    lddt_thresholds = lddt_thresholds.to(dist_diff)
    eps = lddt_thresholds - dist_diff[..., None]
    soft_eps = eps.sigmoid().mean(dim=-1)
    hard_eps = (dist_diff[..., None] < lddt_thresholds).float().mean(dim=-1)

    inclusion_radius = true_dists < cutoff
    # Compute mean, avoiding self term
    mask = inclusion_radius & ~torch.eye(pred_dists.shape[1], dtype=torch.bool, device=pred_dists.device)
    paired_coords_mask = to_pairwise_mask(atom_mask)
    mask = mask & paired_coords_mask

    # Calculate masked averaging
    # _mask = repeat(mask, "... -> ... k", k=len(lddt_thresholds))
    slddt = mask_mean(soft_eps, mask=mask, dim=(-1, -2), eps=1e-4)
    lddt = mask_mean(hard_eps, mask=mask, dim=(-1, -2), eps=1e-4)

    if return_dists:
        return 1.0 - slddt.mean(), lddt.mean(), pred_dists, true_dists
    else:
        return 1.0 - slddt.mean(), lddt.mean()


def compute_rbf(d: torch.Tensor, num_rbf: int = 20, d_min: float = 2.0, d_max: float = 22.0) -> torch.Tensor:
    device = d.device
    shape = d.shape
    bins = torch.linspace(d_min, d_max, num_rbf, device=device)
    sigma = (d_max - d_min) / num_rbf
    d = d.view(-1, 1)
    d = (d - bins[None, :]) / sigma
    x = torch.exp(-(d**2))
    x = x.reshape(shape + (num_rbf,))
    return x


def compute_torsion_angles(plane_points: torch.Tensor, plane_mask=None, return_sincos: bool = False) -> torch.Tensor:
    # points: [..., num_planes, 4, 3]
    a, b, c, d = plane_points.unbind(dim=-2)
    ab = a - b
    bc = b - c
    dc = d - c
    v = normalize(bc)
    n1 = torch.cross(ab, v, dim=-1)
    n2 = torch.cross(dc, v, dim=-1)
    _cos = dotprod(n1, n2)
    _sin = dotprod(torch.cross(n1, n2, dim=-1), v)
    if return_sincos:
        sincos = normalize(torch.stack([_sin, _cos], dim=-1))
        return sincos
    else:
        rad = torch.atan2(_sin, _cos)
        return rad


def compute_backbone_torsions(
    positions: torch.Tensor, is_connected: torch.Tensor, masks: torch.Tensor, use_sincos: bool = True
):
    # positions: [bsz, num_res, 4, 3]
    assert positions.ndim == 4 and positions.shape[-1] == 3
    bsz, num_res = positions.shape[:2]
    # is_connected: [bsz, num_res - 1]. is connected to the next residue
    assert is_connected.shape == (bsz, num_res - 1)
    n_pos, ca_pos, c_pos, o_pos = positions.unbind(dim=-2)
    n_mask, ca_mask, c_mask, o_mask = masks.unbind(dim=-2)
    prev_c_pos, _n_pos, _ca_pos, _c_pos = ca_pos[:, :-1], n_pos[:, 1:], ca_pos[:, 1:], c_pos[:, 1:]
    prev_c_mask, _n_mask, _ca_mask, _c_mask = ca_mask[:, :-1], n_mask[:, 1:], ca_mask[:, 1:], c_mask[:, 1:]
    _phi_angle = compute_torsion_angles(
        torch.stack([prev_c_pos, _n_pos, _ca_pos, _c_pos], dim=-2), return_sincos=use_sincos
    )
    _phi_mask = prev_c_mask & _n_mask & _ca_mask & _c_mask & is_connected
    if not use_sincos:
        _phi_angle.masked_fill_(~_phi_mask, 0.0)
        _phi_angle = F.pad(_phi_angle, (1, 0, 0, 0), value=0.0)  # [bsz, num_res]
    else:
        _phi_angle = _phi_angle.masked_fill(~_phi_mask[..., None], 0.0)
        _phi_angle = F.pad(_phi_angle, (0, 0, 1, 0, 0, 0), value=0.0)

    _n_pos, _ca_pos, _c_pos, next_n_pos = n_pos[:, :-1], ca_pos[:, :-1], c_pos[:, :-1], n_pos[:, 1:]
    _n_mask, _ca_mask, _c_mask, next_n_mask = n_mask[:, :-1], ca_mask[:, :-1], c_mask[:, :-1], n_mask[:, 1:]
    _psi_angle = compute_torsion_angles(
        torch.stack([_n_pos, _ca_pos, _c_pos, next_n_pos], dim=-2), return_sincos=use_sincos
    )
    _psi_mask = _n_mask & _ca_mask & _c_mask & next_n_mask & is_connected
    if not use_sincos:
        _psi_angle.masked_fill_(~_psi_mask, 0.0)
        _psi_angle = F.pad(_psi_angle, (0, 1, 0, 0), value=0.0)
    else:
        _psi_angle = _psi_angle.masked_fill(~_psi_mask[..., None], 0.0)
        _psi_angle = F.pad(_psi_angle, (0, 0, 0, 1, 0, 0), value=0.0)

    _ca_pos, _c_pos, next_ca_pos, next_n_pos = ca_pos[:, :-1], c_pos[:, :-1], ca_pos[:, 1:], n_pos[:, 1:]
    _ca_mask, _c_mask, next_ca_mask, next_n_mask = ca_mask[:, :-1], c_mask[:, :-1], ca_mask[:, 1:], n_mask[:, 1:]
    _omega_angle = compute_torsion_angles(
        torch.stack([_ca_pos, _c_pos, next_n_pos, next_ca_pos], dim=-2), return_sincos=use_sincos
    )
    _omega_mask = _ca_mask & _c_mask & next_n_mask & next_ca_mask & is_connected
    if not use_sincos:
        _omega_angle.masked_fill_(~_omega_mask, 0.0)
        _omega_angle = F.pad(_omega_angle, (0, 1, 0, 0), value=0.0)
    else:
        _omega_angle = _omega_angle.masked_fill(~_omega_mask[..., None], 0.0)
        _omega_angle = F.pad(_omega_angle, (0, 0, 0, 1, 0, 0), value=0.0)

    if use_sincos:
        return torch.concat((_phi_angle, _psi_angle, _omega_angle), dim=-1)
    else:
        return torch.stack((_phi_angle, _psi_angle, _omega_angle), dim=-1)


@torch.autocast(device_type="cuda", enabled=False)
def calculate_dihedral(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor, eps: float = 1e-8):
    p1, p2, p3, p4 = map(lambda x: x.to(dtype=torch.float32), (p1, p2, p3, p4))
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    # Robust normalization
    n1_norm = norm(n1, dim=-1, keepdim=True, eps=eps)
    n2_norm = norm(n2, dim=-1, keepdim=True, eps=eps)

    n1 = n1 / torch.clamp(n1_norm, min=eps)
    n2 = n2 / torch.clamp(n2_norm, min=eps)

    b2_norm = norm(b2, dim=-1, keepdim=True)
    b2_unit = b2 / torch.clamp(b2_norm, min=eps)

    cos_angle = torch.sum(n1 * n2, dim=-1)
    sin_angle = torch.sum(torch.cross(n1, n2, dim=-1) * b2_unit, dim=-1)

    return torch.stack([cos_angle, sin_angle], dim=-1)


def norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp(torch.sum(x**2, dim=dim, keepdim=keepdim), min=eps))


@torch.autocast(enabled=False, device_type="cuda")
def compute_rbf(d: torch.Tensor, rbf_mu: torch.Tensor) -> torch.Tensor:
    d = d.float()
    rbf = torch.exp(-(d.unsqueeze(-1) - rbf_mu).square())
    return rbf


@torch.autocast(enabled=False, device_type="cuda")
def compute_disto(d: torch.Tensor, disto_bins: torch.Tensor) -> torch.Tensor:
    d = d.float()
    d_labels = to_one_hot((d.unsqueeze(-1) >= disto_bins).sum(-1), len(disto_bins) + 2)
    return d_labels


@torch.autocast(enabled=False, device_type="cuda")
def compute_pairwise_torsion(pos: torch.Tensor, mask=None):
    # pos should be of shape (b, n, k, 3)
    k = pos.shape[1]
    n, ca, c = map(lambda x: repeat(x, "b n d -> b n k d", k=k), [pos[..., 0, :], pos[..., 1, :], pos[..., 2, :]])
    ca_other = repeat(pos[..., 1, :], "b n d -> b k n d", k=k)
    pairwise_torsion = calculate_dihedral(n, ca, c, ca_other)
    if mask is not None:
        col_mask = repeat(torch.prod(mask[..., :3], dim=-1), "b n -> b n k", k=k)
        row_mask = repeat(mask[..., 1], "b n -> b k n", k=k)
        pairwise_mask = col_mask * row_mask
        pairwise_torsion = pairwise_torsion.masked_fill(~pairwise_mask[..., None].bool(), 0.0)
    return pairwise_torsion


@torch.autocast(enabled=False, device_type="cuda")
def compute_atom_dist(pos: torch.Tensor, mask=None, eps=1e-6, fill_value=0):
    """Computer all atom pairwise distances"""
    if pos.ndim == 3:
        # pos: (b, n, 3)
        dist = ((pos[:, None, :] - pos[:, :, None]).square().sum(dim=-1) + eps).sqrt()
    else:
        # pos: (b, n, k, 3)
        dist = ((pos[:, None, :, None, :] - pos[:, :, None, :, None]).square().sum(dim=-1) + eps).sqrt()
        if mask is not None:
            dist_mask = (mask[:, None, :, None, :] * mask[:, :, None, :, None]).bool()
            dist = dist.masked_fill(~dist_mask, fill_value)
        dist = rearrange(dist, "b n m k l -> b n m (k l)")
    return dist
