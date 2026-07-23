from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from unilab.algos.torch.amp.motion_dataset import WalkMotionDataset
from unilab.algos.torch.amp.spec import AMP_OBSERVATION_DIM, build_amp_observation

_ASSET_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "unilab"
    / "assets"
    / "motions"
    / "g1"
    / "amp_walk"
)
_MANIFEST = _ASSET_ROOT / "manifest.json"


def test_manifest_is_explicitly_forward_only() -> None:
    dataset = WalkMotionDataset.from_manifest(_MANIFEST)

    assert dataset.motion_names == (
        "walk_forward_loop_002__A022",
        "walk_forward_loop_002__A024",
    )
    assert dataset.num_transitions == (455 - 1) + (482 - 1)
    assert AMP_OBSERVATION_DIM == 195


def test_manifest_rejects_non_forward_motion_before_file_access(tmp_path: Path) -> None:
    manifest = json.loads(_MANIFEST.read_text())
    manifest["motions"][0]["file"] = "jog_forward_loop_003__A021.npz"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="forward-walk naming contract"):
        WalkMotionDataset.from_manifest(path, asset_root=_ASSET_ROOT)


def test_manifest_rejects_asset_hash_mismatch(tmp_path: Path) -> None:
    manifest = json.loads(_MANIFEST.read_text())
    manifest["motions"][0]["sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="SHA-256"):
        WalkMotionDataset.from_manifest(path, asset_root=_ASSET_ROOT)


def test_sampling_is_deterministic_and_never_crosses_clip_boundaries() -> None:
    dataset = WalkMotionDataset.from_manifest(_MANIFEST)

    first = dataset.sample(128, seed=20260722)
    second = dataset.sample(128, seed=20260722)

    np.testing.assert_array_equal(first.current, second.current)
    np.testing.assert_array_equal(first.next, second.next)
    np.testing.assert_array_equal(first.motion_index, second.motion_index)
    np.testing.assert_array_equal(first.frame_index, second.frame_index)
    for row, (motion_index, frame_index) in enumerate(
        zip(first.motion_index, first.frame_index, strict=True)
    ):
        features = dataset.motion_features[int(motion_index)]
        np.testing.assert_array_equal(first.current[row], features[int(frame_index)])
        np.testing.assert_array_equal(first.next[row], features[int(frame_index) + 1])


def test_first_source_frame_matches_amp_mjlab_feature_oracle() -> None:
    source = np.load(_ASSET_ROOT / "walk_forward_loop_002__A022.npz")
    feature = build_amp_observation(
        body_pos_w=source["body_pos_w"][:1],
        body_quat_w=source["body_quat_w"][:1],
        body_lin_vel_w=source["body_lin_vel_w"][:1],
        body_ang_vel_w=source["body_ang_vel_w"][:1],
    )[0]

    expected_head = np.array(
        [
            0.0117782503,
            0.0000487077,
            -0.0425790809,
            0.0299827307,
            0.1165473685,
            -0.1743387431,
            0.0555004701,
            0.1117153168,
            -0.4794005454,
            0.0350014120,
            0.1007262468,
            -0.7957944870,
        ],
        dtype=np.float32,
    )
    expected_boundaries = np.array(
        [
            -0.1061289757,
            0.9835679531,
            -0.0230260286,
            -0.1581987143,
            -0.0086476393,
            -0.0067800623,
            -0.0116134658,
            -0.0031881798,
            0.0050200894,
        ],
        dtype=np.float32,
    )

    assert feature.shape == (195,)
    np.testing.assert_allclose(feature[:12], expected_head, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        feature[[38, 39, 40, 116, 117, 118, 155, 156, 194]],
        expected_boundaries,
        rtol=1e-6,
        atol=1e-6,
    )
    assert float(feature.sum()) == pytest.approx(18.4473114, abs=1e-5)
