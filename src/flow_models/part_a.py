"""Model-agnostic sampling and image-composition methods for Part A.

Tensors use BCHW layout and the assignment convention t=0 (noise), t=1 (image).
The callable `velocity(x, t)` must return a velocity tensor matching x.
"""
from __future__ import annotations

from collections.abc import Callable
import torch
from torch import Tensor


def interpolate(image: Tensor, noise: Tensor, t: float | Tensor) -> Tensor:
    """Linear OT path between noise and image: x_t=(1-t)noise+t image."""
    if image.shape != noise.shape:
        raise ValueError(f"image and noise shapes differ: {image.shape} vs {noise.shape}")
    time = torch.as_tensor(t, device=image.device, dtype=image.dtype)
    if torch.any((time < 0) | (time > 1)):
        raise ValueError("t must lie in [0, 1]")
    while time.ndim < image.ndim:
        time = time.unsqueeze(-1)
    return (1 - time) * noise + time * image


def one_step_denoise(x_t: Tensor, t: float, velocity: Callable[[Tensor, float], Tensor]) -> Tensor:
    """Estimate x_1 from one velocity evaluation: x_t + (1-t)v(x_t,t)."""
    if not 0 <= t <= 1:
        raise ValueError("t must lie in [0, 1]")
    return x_t + (1 - t) * velocity(x_t, t)


@torch.no_grad()
def euler_sample(
    x_start: Tensor,
    t_start: float,
    step_size: float,
    velocity: Callable[[Tensor, float], Tensor],
    *,
    t_end: float = 1.0,
) -> tuple[Tensor, list[tuple[float, Tensor]]]:
    """Integrate dx/dt=v(x,t), shortening the final step to land exactly at t_end.

    Returns the final tensor and snapshots after each update (time, detached state).
    """
    if not 0 <= t_start <= t_end <= 1:
        raise ValueError("require 0 <= t_start <= t_end <= 1")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    x = x_start.float()
    t = float(t_start)
    states: list[tuple[float, Tensor]] = [(t, x.detach().clone())]
    while t < t_end - 1e-8:
        h = min(step_size, t_end - t)
        v = velocity(x, t)
        if v.shape != x.shape:
            raise ValueError(f"velocity shape {v.shape} does not match state {x.shape}")
        x = x + h * v.float()
        t = min(t + h, t_end)
        states.append((t, x.detach().clone()))
    return x, states


def cfg_velocity(
    x: Tensor,
    t: float,
    predict: Callable[[Tensor, float, object], Tensor],
    conditional: object,
    unconditional: object,
    weight: float = 7.0,
) -> Tensor:
    """Classifier-free guidance using v_c + w(v_c-v_u)."""
    v_cond = predict(x, t, conditional)
    if weight == 0:
        return v_cond
    v_uncond = predict(x, t, unconditional)
    if v_cond.shape != v_uncond.shape:
        raise ValueError("conditional and unconditional velocity shapes differ")
    return v_cond + weight * (v_cond - v_uncond)


def visual_anagram_velocity(
    x: Tensor, t: float, predict_cfg: Callable[[Tensor, float, object], Tensor], prompt_a: object, prompt_b: object
) -> Tensor:
    """Average CFG velocity in upright and 180-degree-rotated views."""
    v_a = predict_cfg(x, t, prompt_a)
    rotated = torch.flip(x, dims=(-2, -1))
    v_b = torch.flip(predict_cfg(rotated, t, prompt_b), dims=(-2, -1))
    return (v_a + v_b) * 0.5


def hybrid_velocity(
    x: Tensor,
    t: float,
    predict_cfg: Callable[[Tensor, float, object], Tensor],
    near_prompt: object,
    far_prompt: object,
    blur: Callable[[Tensor], Tensor],
) -> Tensor:
    """Combine low frequencies from near_prompt with high frequencies from far_prompt."""
    v_near = predict_cfg(x, t, near_prompt)
    v_far = predict_cfg(x, t, far_prompt)
    low = blur(v_near)
    high = v_far - blur(v_far)
    return low + high


def edit_image(
    image: Tensor,
    noise: Tensor,
    start_t: float,
    velocity: Callable[[Tensor, float], Tensor],
    step_size: float = 0.02,
) -> Tensor:
    """SDEdit-style image edit: corrupt to start_t, then integrate to t=1."""
    return euler_sample(interpolate(image, noise, start_t), start_t, step_size, velocity)[0]
