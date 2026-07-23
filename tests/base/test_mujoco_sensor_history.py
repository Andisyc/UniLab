from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.base.backend.base import SimBackend


class _FakePool:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def step(
        self,
        state,
        *,
        nstep,
        control,
        control_spec,
        return_sensor=False,
        post_step_forward_sensor=False,
    ):
        del control, control_spec, post_step_forward_sensor
        self.calls.append(int(nstep))
        state_out = np.asarray(state) + float(nstep)
        sensor_out = state_out[:, :1]
        return (state_out, sensor_out) if return_sensor else state_out


def _fake_backend():
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    backend = object.__new__(MuJoCoBackend)
    backend._pre_step_control_fn = None
    backend._num_envs = 1
    backend._np_dtype = np.float32
    backend._physics_state = np.zeros((1, 11), dtype=np.float32)
    backend._sensor_data = np.zeros((1, 1), dtype=np.float32)
    backend._sensor_indices = {"contact": [0]}
    backend._sensor_histories = {}
    backend._sensor_history_specs = {}
    backend._sensor_ids = {"contact": 0}
    backend._sensor_history_state_offset = 1
    backend._model = SimpleNamespace(
        sensor_history=np.array([[4, 0]], dtype=np.int32),
        sensor_historyadr=np.array([0], dtype=np.int32),
        sensor_dim=np.array([1], dtype=np.int32),
    )
    backend._pending_xfrc_applied = np.zeros((1, 0), dtype=np.float64)
    backend._post_step_forward_sensor = False
    backend._pool = _FakePool()
    return backend


def test_sim_backend_sensor_history_fails_closed_by_default() -> None:
    backend = SimpleNamespace()

    with pytest.raises(NotImplementedError, match="sensor history"):
        SimBackend.configure_sensor_history(backend, "contact", history_length=4)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="sensor history"):
        SimBackend.get_sensor_history(backend, "contact")  # type: ignore[arg-type]


def test_mujoco_sensor_history_is_opt_in_and_preserves_default_fast_path() -> None:
    backend = _fake_backend()
    ctrl = np.zeros((1, 0), dtype=np.float32)

    backend.step(ctrl, nsteps=3)

    assert backend._pool.calls == [3]


def test_mujoco_sensor_history_reads_native_ring_oldest_to_newest() -> None:
    backend = _fake_backend()
    backend.configure_sensor_history("contact", history_length=4)
    # Native layout: metadata, timestamps, then sample-major values. Latest=2.
    backend._physics_state[0] = [0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0]
    backend._refresh_sensor_histories_from_state()

    history = backend.get_sensor_history("contact")
    assert history.shape == (1, 4, 1)
    np.testing.assert_array_equal(history[0, :, 0], [40.0, 10.0, 20.0, 30.0])


def test_mujoco_sensor_history_rejects_missing_or_conflicting_configuration() -> None:
    backend = _fake_backend()

    with pytest.raises(ValueError, match="positive"):
        backend.configure_sensor_history("contact", history_length=0)
    with pytest.raises(ValueError, match="not found"):
        backend.configure_sensor_history("missing", history_length=4)

    with pytest.raises(ValueError, match="native history length"):
        backend.configure_sensor_history("contact", history_length=3)

    backend.configure_sensor_history("contact", history_length=4)
    backend.configure_sensor_history("contact", history_length=4)


@pytest.mark.slow
def test_mujoco_real_contact_history_is_nonzero_and_partial_reset_isolated(tmp_path) -> None:
    from unilab.base.backend.mujoco.backend import MuJoCoBackend
    from unilab.base.scene import SceneCfg

    model_file = tmp_path / "self_contact.xml"
    model_file.write_text(
        """
<mujoco>
  <option gravity="0 0 0" timestep="0.005"/>
  <worldbody>
    <body name="root">
      <freejoint/>
      <geom name="root_inertia" type="sphere" size="0.01" mass="0.1"
        contype="0" conaffinity="0"/>
      <body name="left">
        <joint name="left_joint" type="hinge" axis="0 1 0"/>
        <geom name="left_geom" type="sphere" size="0.1" pos="0 0 0"/>
      </body>
      <body name="right">
        <joint name="right_joint" type="hinge" axis="1 0 0"/>
        <geom name="right_geom" type="sphere" size="0.1" pos="0.15 0 0"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <contact name="self_collision" subtree1="root" subtree2="root"
      data="force" num="1" reduce="none" nsample="4"/>
  </sensor>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=2,
        sim_dt=0.005,
        base_name="root",
    )
    backend.configure_sensor_history("self_collision", history_length=4)
    backend.materialize()
    try:
        backend.step(np.zeros((2, 0), dtype=np.float32), nsteps=4)
        before = backend.get_sensor_history("self_collision").copy()
        assert before.shape == (2, 4, 3)
        assert np.isfinite(before).all()
        assert np.max(np.linalg.norm(before, axis=-1)) > 0.0

        qpos = np.broadcast_to(backend.model.qpos0, (1, backend.model.nq)).copy()
        qvel = np.zeros((1, backend.model.nv), dtype=np.float64)
        backend.set_state(np.array([0], dtype=np.int32), qpos, qvel)
        after = backend.get_sensor_history("self_collision")

        np.testing.assert_array_equal(after[0], 0.0)
        np.testing.assert_array_equal(after[1], before[1])
    finally:
        if backend._pool is not None:
            backend._pool.close()
        backend.cleanup_scene_assets()


@pytest.mark.slow
def test_mujoco_sensor_history_does_not_change_physics_trajectory(tmp_path) -> None:
    from unilab.base.backend.mujoco.backend import MuJoCoBackend
    from unilab.base.scene import SceneCfg

    model_file = tmp_path / "trajectory_contact.xml"
    model_file.write_text(
        """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005" iterations="10"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="root" pos="0 0 0.3">
      <freejoint/>
      <geom name="root_geom" type="sphere" size="0.1" mass="1"/>
      <body name="arm" pos="0.15 0 0">
        <joint name="arm_joint" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="arm_geom" type="capsule" size="0.04 0.12" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <contact name="self_collision" subtree1="root" subtree2="root"
      data="force" num="1" reduce="none" nsample="4"/>
  </sensor>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    scene = SceneCfg(model_file=str(model_file))
    baseline = MuJoCoBackend(scene, num_envs=2, sim_dt=0.005, base_name="root")
    instrumented = MuJoCoBackend(scene, num_envs=2, sim_dt=0.005, base_name="root")
    instrumented.configure_sensor_history("self_collision", history_length=4)
    baseline.materialize()
    instrumented.materialize()
    try:
        ctrl = np.zeros((2, 0), dtype=np.float32)
        for _ in range(20):
            baseline.step(ctrl, nsteps=4)
            instrumented.step(ctrl, nsteps=4)

        np.testing.assert_array_equal(
            instrumented.get_physics_state(),
            baseline.get_physics_state(),
        )
    finally:
        for backend in (baseline, instrumented):
            if backend._pool is not None:
                backend._pool.close()
            backend.cleanup_scene_assets()
