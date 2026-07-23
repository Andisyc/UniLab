"""Canonical G1 AMP observation contract shared by expert and policy states."""

from __future__ import annotations

import numpy as np

from unilab.envs.common.rotation import (
    np_matrix_first_two_cols_from_quat,
    np_quat_apply_batched,
    np_quat_conjugate_batched,
    np_subtract_anchor_frame_transforms,
)

AMP_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
AMP_BODY_INDICES = (0, 2, 4, 6, 8, 10, 12, 17, 19, 22, 24, 26, 29)
AMP_ANCHOR_BODY_NAME = "torso_link"
AMP_ANCHOR_BODY_INDEX = 15
AMP_FEATURES_PER_BODY = 3 + 6 + 3 + 3
AMP_OBSERVATION_DIM = len(AMP_BODY_NAMES) * AMP_FEATURES_PER_BODY


def _validate_body_state(name: str, value: np.ndarray, width: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != width:
        raise ValueError(f"{name} must have shape [batch, body, {width}], got {array.shape}")
    if array.shape[1] <= max(max(AMP_BODY_INDICES), AMP_ANCHOR_BODY_INDEX):
        raise ValueError(
            f"{name} must contain at least 30 ordered G1 bodies, got {array.shape[1]}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def build_amp_observation(
    *,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    body_ang_vel_w: np.ndarray,
) -> np.ndarray:
    """Build the AMP_mjlab-compatible 195-D state for ordered G1 body arrays.

    The concatenation order is position, 6-D orientation, local linear
    velocity, then local angular velocity. Each term is flattened over the
    canonical 13-body order.
    """
    body_pos_w = _validate_body_state("body_pos_w", body_pos_w, 3)
    body_quat_w = _validate_body_state("body_quat_w", body_quat_w, 4)
    body_lin_vel_w = _validate_body_state("body_lin_vel_w", body_lin_vel_w, 3)
    body_ang_vel_w = _validate_body_state("body_ang_vel_w", body_ang_vel_w, 3)
    batch_shape = body_pos_w.shape[:2]
    for name, value in (
        ("body_quat_w", body_quat_w),
        ("body_lin_vel_w", body_lin_vel_w),
        ("body_ang_vel_w", body_ang_vel_w),
    ):
        if value.shape[:2] != batch_shape:
            raise ValueError(
                f"{name} batch/body shape {value.shape[:2]} does not match {batch_shape}"
            )

    body_indices = np.asarray(AMP_BODY_INDICES, dtype=np.intp)
    return build_amp_observation_from_selected(
        body_pos_w=body_pos_w[:, body_indices],
        body_quat_w=body_quat_w[:, body_indices],
        body_lin_vel_w=body_lin_vel_w[:, body_indices],
        body_ang_vel_w=body_ang_vel_w[:, body_indices],
        anchor_pos_w=body_pos_w[:, AMP_ANCHOR_BODY_INDEX],
        anchor_quat_w=body_quat_w[:, AMP_ANCHOR_BODY_INDEX],
    )


def build_amp_observation_from_selected(
    *,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    body_ang_vel_w: np.ndarray,
    anchor_pos_w: np.ndarray,
    anchor_quat_w: np.ndarray,
) -> np.ndarray:
    """Build 195-D AMP state from canonical 13-body arrays plus torso anchor."""
    expected = len(AMP_BODY_NAMES)
    arrays = {
        "body_pos_w": (np.asarray(body_pos_w), 3),
        "body_quat_w": (np.asarray(body_quat_w), 4),
        "body_lin_vel_w": (np.asarray(body_lin_vel_w), 3),
        "body_ang_vel_w": (np.asarray(body_ang_vel_w), 3),
    }
    batch_size: int | None = None
    for name, (value, width) in arrays.items():
        if value.ndim != 3 or value.shape[1:] != (expected, width):
            raise ValueError(
                f"{name} must have shape [batch, {expected}, {width}], got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
        if batch_size is None:
            batch_size = value.shape[0]
        elif value.shape[0] != batch_size:
            raise ValueError(f"{name} batch size {value.shape[0]} does not match {batch_size}")
    assert batch_size is not None
    anchor_pos_w = np.asarray(anchor_pos_w)
    anchor_quat_w = np.asarray(anchor_quat_w)
    if anchor_pos_w.shape != (batch_size, 3):
        raise ValueError(f"anchor_pos_w must have shape [{batch_size}, 3], got {anchor_pos_w.shape}")
    if anchor_quat_w.shape != (batch_size, 4):
        raise ValueError(
            f"anchor_quat_w must have shape [{batch_size}, 4], got {anchor_quat_w.shape}"
        )

    relative_pos, relative_quat = np_subtract_anchor_frame_transforms(
        anchor_pos_w,
        anchor_quat_w,
        body_pos_w,
        body_quat_w,
    )
    orientation_6d = np_matrix_first_two_cols_from_quat(relative_quat)
    inverse_body_quat = np_quat_conjugate_batched(body_quat_w)
    local_lin_vel = np_quat_apply_batched(
        inverse_body_quat,
        body_lin_vel_w,
    )
    local_ang_vel = np_quat_apply_batched(
        inverse_body_quat,
        body_ang_vel_w,
    )

    features = np.concatenate(
        (
            relative_pos.reshape(batch_size, -1),
            orientation_6d.reshape(batch_size, -1),
            local_lin_vel.reshape(batch_size, -1),
            local_ang_vel.reshape(batch_size, -1),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    if features.shape != (batch_size, AMP_OBSERVATION_DIM):
        raise RuntimeError(f"invalid AMP feature shape {features.shape}")
    return np.ascontiguousarray(features)
