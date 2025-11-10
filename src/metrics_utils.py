import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import lpips  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    lpips = None  # type: ignore


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Compute PSNR on tensors in [0, max_val]."""
    target = target.to(dtype=pred.dtype)
    mse = F.mse_loss(pred, target, reduction="mean")
    if mse == 0:
        return torch.tensor(float("inf"), device=pred.device)
    psnr = 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse)
    return psnr


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device, channel: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_1d = gauss.unsqueeze(1)
    kernel_2d = kernel_1d @ kernel_1d.t()
    kernel = kernel_2d.expand(channel, 1, window_size, window_size).contiguous()
    return kernel


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Fast SSIM implementation supporting multi-channel tensors."""
    target = target.to(dtype=pred.dtype)
    channel = pred.size(1)
    kernel = _gaussian_kernel(window_size, sigma, pred.device, channel)
    kernel = kernel.to(dtype=pred.dtype)

    mu1 = F.conv2d(pred, kernel, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, kernel, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


class LPIPSHelper:
    """Lazy LPIPS metric loader."""

    def __init__(self, net: str = "alex", device: Optional[torch.device] = None):
        self.net = net
        self.device = device
        self.metric = None

    def _load(self):
        if self.metric is None:
            if lpips is None:
                raise RuntimeError("LPIPS requested but lpips library is not installed.")
            self.metric = lpips.LPIPS(net=self.net).to(self.device or torch.device("cpu"))
            self.metric.eval()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self._load()
        with torch.no_grad():
            if self.metric is None:
                raise RuntimeError("LPIPS metric failed to initialize.")
            return self.metric(pred, target).mean()


def semantic_cosine_similarity(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between semantic feature tensors."""
    a = feat_a.flatten(2)
    b = feat_b.flatten(2)
    sim = F.cosine_similarity(a, b, dim=2)
    return sim.mean()


def semantic_ssim(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    """Compute SSIM over semantic features."""
    # Normalize features to [0, 1] per channel to stabilise SSIM.
    def _norm(feat: torch.Tensor) -> torch.Tensor:
        flat = feat.flatten(2)
        min_val = flat.min(dim=2, keepdim=True)[0].unsqueeze(2)
        max_val = flat.max(dim=2, keepdim=True)[0].unsqueeze(2)
        norm = (feat - min_val) / (max_val - min_val + 1e-6)
        return torch.clamp(norm, 0.0, 1.0)

    feat_a_norm = _norm(feat_a)
    feat_b_norm = _norm(feat_b)
    return compute_ssim(feat_a_norm, feat_b_norm)


def compute_iter_speed(iter_time_s: float, batch_size: int) -> Tuple[float, float]:
    """Return milliseconds per iteration and images / second."""
    iter_ms = iter_time_s * 1000.0
    speed = batch_size / iter_time_s if iter_time_s > 0 else 0.0
    return iter_ms, speed


_NIQE_METRIC = None
_BRISQUE_METRIC = None


def maybe_compute_niqe(images: torch.Tensor) -> Optional[torch.Tensor]:
    """Compute NIQE if optional dependency is present."""
    global _NIQE_METRIC
    try:
        import piq  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        warnings.warn("compute_niqe_brisque=True but piq is not installed. Skipping NIQE.", RuntimeWarning)
        return None

    if _NIQE_METRIC is None:
        try:
            _NIQE_METRIC = piq.NIQE().to(images.device)
        except AttributeError:
            warnings.warn("Installed piq does not expose NIQE metric. Skipping NIQE.", RuntimeWarning)
            return None

    with torch.no_grad():
        return _NIQE_METRIC(images).mean()


def maybe_compute_brisque(images: torch.Tensor) -> Optional[torch.Tensor]:
    """Compute BRISQUE if optional dependency is present."""
    global _BRISQUE_METRIC
    try:
        import piq  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        warnings.warn("compute_niqe_brisque=True but piq is not installed. Skipping BRISQUE.", RuntimeWarning)
        return None

    if _BRISQUE_METRIC is None:
        try:
            _BRISQUE_METRIC = piq.BRISQUELoss().to(images.device)
        except AttributeError:
            warnings.warn("Installed piq does not expose BRISQUE metric. Skipping BRISQUE.", RuntimeWarning)
            return None

    with torch.no_grad():
        # BRISQUELoss returns higher-worse, same as standard BRISQUE score
        return _BRISQUE_METRIC(images).mean()


def _rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor to grayscale using luminance weights."""
    if image.size(1) == 1:
        return image
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def compute_halo_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_threshold: float = 0.03,
    blur_kernel: int = 7
) -> torch.Tensor:
    """
    Approximate halo strength by measuring excess brightening along high-frequency edges.

    A higher value indicates stronger halo artifacts. Returns the average halo score over the batch.
    """
    if blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be odd.")

    pred_gray = _rgb_to_gray(pred)
    target_gray = _rgb_to_gray(target.to(dtype=pred.dtype, device=pred.device))

    lap_kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=pred.device,
        dtype=pred.dtype
    ).view(1, 1, 3, 3)

    edge_mag = torch.abs(F.conv2d(target_gray, lap_kernel, padding=1))
    edge_mask = (edge_mag > edge_threshold).to(pred.dtype)

    blur = torch.ones((1, 1, blur_kernel, blur_kernel), device=pred.device, dtype=pred.dtype)
    blur = blur / blur.numel()
    pred_blur = F.conv2d(pred_gray, blur, padding=blur_kernel // 2)
    target_blur = F.conv2d(target_gray, blur, padding=blur_kernel // 2)

    high_freq_pred = pred_gray - pred_blur
    high_freq_target = target_gray - target_blur

    halo_activation = torch.relu(high_freq_pred - high_freq_target)
    halo_score = (halo_activation * edge_mask).mean()
    return halo_score


def _srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    """Convert sRGB gamma-corrected values to linear RGB."""
    threshold = 0.04045
    below = image / 12.92
    above = ((image + 0.055) / 1.055).clamp(min=0) ** 2.4
    return torch.where(image <= threshold, below, above)


def _rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image in [0, 1] range to LAB color space (D65 reference).
    """
    if image.size(1) != 3:
        raise ValueError("RGB to LAB conversion expects 3-channel input.")

    linear = _srgb_to_linear(image)
    rgb_to_xyz = torch.tensor(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        device=image.device,
        dtype=image.dtype
    )
    linear_flat = linear.flatten(2)
    xyz = torch.matmul(rgb_to_xyz, linear_flat)
    xyz = xyz.view_as(linear)

    xyz_ref = torch.tensor([0.95047, 1.00000, 1.08883], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    xyz_norm = xyz / xyz_ref

    delta = 6 / 29
    def _f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(
            t > delta ** 3,
            t.pow(1.0 / 3.0),
            t / (3 * delta ** 2) + 4.0 / 29.0
        )

    f_x = _f(xyz_norm[:, 0:1, ...])
    f_y = _f(xyz_norm[:, 1:2, ...])
    f_z = _f(xyz_norm[:, 2:3, ...])

    L = 116.0 * f_y - 16.0
    a = 500.0 * (f_x - f_y)
    b = 200.0 * (f_y - f_z)
    return torch.cat([L, a, b], dim=1)


def compute_delta_e(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute mean Delta E (CIE76) between prediction and target.

    Lower values indicate closer color reproduction.
    """
    target = target.to(dtype=pred.dtype, device=pred.device)
    pred_lab = _rgb_to_lab(pred)
    target_lab = _rgb_to_lab(target)
    delta = pred_lab - target_lab
    delta_e = torch.sqrt(torch.clamp((delta ** 2).sum(dim=1, keepdim=True), min=1e-12))
    return delta_e.mean()
