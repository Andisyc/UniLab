"""Isolated fixed-forward G1 environment contract for AMP Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.algos.torch.amp.spec import (
    AMP_ANCHOR_BODY_NAME,
    AMP_BODY_NAMES,
    AMP_OBSERVATION_DIM,
    build_amp_observation_from_selected,
)
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.g1.joystick import G1WalkEnv, G1WalkFlatCfg

_GAIT_PHASE_REWARD_TERMS = frozenset(
    {"feet_phase", "feet_phase_contrast", "feet_phase_contact", "feet_double_stance"}
)


def _fixed_forward_commands() -> Commands:
    return Commands(
        vel_limit=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        rel_standing_envs=0.0,
        rel_transition_envs=0.0,
        small_xy_threshold=0.0,
        heading_command=False,
        observe_height_command=False,
        random_height_during_walking=False,
    )


def _amp_walk_scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"),
        fragment_files=[str(ASSETS_ROOT_PATH / "robots" / "g1" / "amp_walk_task.xml")],
    )


def compute_self_collision_cost(
    force_history: np.ndarray, *, force_threshold: float
) -> np.ndarray:
    """Count history entries where any contact slot exceeds the force threshold."""
    history = np.asarray(force_history)
    if history.ndim != 3 or history.shape[-1] % 3 != 0:
        raise ValueError(
            "force_history must have shape (num_envs, history_length, 3 * num_slots), "
            f"got {history.shape}"
        )
    slot_forces = history.reshape(history.shape[0], history.shape[1], -1, 3)
    force_magnitude = np.linalg.norm(slot_forces, axis=-1)
    hit = np.any(force_magnitude > float(force_threshold), axis=-1)
    return np.asarray(np.sum(hit, axis=-1), dtype=np.float32)


@registry.envcfg("G1AMPWalk")
@dataclass
class G1AMPWalkCfg(G1WalkFlatCfg):
    scene: SceneCfg = field(default_factory=_amp_walk_scene)
    commands: Commands = field(default_factory=_fixed_forward_commands)
    mode_observation: bool = False
    add_body_sensors: bool = True
    self_collision_sensor_name: str = "self_collision"
    self_collision_history_length: int = 4
    self_collision_force_threshold: float = 10.0


class G1AMPWalkEnv(G1WalkEnv):
    """G1 walk task with a canonical AMP group and no gait-phase observation."""

    def __init__(self, cfg: G1AMPWalkCfg, num_envs=1, backend_type="mujoco"):
        self._validate_phase1_cfg(cfg)
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._backend.configure_sensor_history(
            cfg.self_collision_sensor_name,
            history_length=cfg.self_collision_history_length,
        )
        names = (*AMP_BODY_NAMES, AMP_ANCHOR_BODY_NAME)
        self._amp_body_ids_with_anchor = self._backend.get_body_ids(names)

    @staticmethod
    def _validate_phase1_cfg(cfg: G1AMPWalkCfg) -> None:
        expected = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        if not np.array_equal(np.asarray(cfg.commands.vel_limit), expected):
            raise ValueError("G1AMPWalk requires fixed command [1.0, 0.0, 0.0]")
        if cfg.commands.rel_standing_envs != 0.0 or cfg.commands.rel_transition_envs != 0.0:
            raise ValueError("G1AMPWalk does not allow standing or transition command sampling")
        if cfg.commands.heading_command:
            raise ValueError("G1AMPWalk does not allow heading control")
        if cfg.mode_observation or cfg.commands.observe_height_command:
            raise ValueError("G1AMPWalk does not allow mode or height observations")
        scales = getattr(cfg.reward_config, "scales", {}) or {}
        if float(scales.get("pose", 0.0)) != 0.0:
            raise ValueError("G1AMPWalk does not allow default-pose reward authority")
        enabled_gait_terms = sorted(
            term for term in _GAIT_PHASE_REWARD_TERMS if float(scales.get(term, 0.0)) != 0.0
        )
        if enabled_gait_terms:
            raise ValueError(
                "G1AMPWalk does not allow gait-phase rewards: " + ", ".join(enabled_gait_terms)
            )
        if float(scales.get("self_collisions", 0.0)) != -0.1:
            raise ValueError("G1AMPWalk requires self_collisions reward scale -0.1")
        if cfg.self_collision_sensor_name != "self_collision":
            raise ValueError("G1AMPWalk requires the self_collision sensor")
        if cfg.self_collision_history_length != 4:
            raise ValueError("G1AMPWalk requires self-collision history length 4")
        if cfg.self_collision_force_threshold != 10.0:
            raise ValueError("G1AMPWalk requires self-collision force threshold 10.0 N")

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns["self_collisions"] = self._reward_self_collisions

    def _reward_self_collisions(self, ctx) -> np.ndarray:
        history = self._backend.get_sensor_history(self._cfg.self_collision_sensor_name)
        cost = compute_self_collision_cost(
            history,
            force_threshold=self._cfg.self_collision_force_threshold,
        )
        ctx.info["self_collision_hit_count"] = cost
        ctx.info["self_collision_hit_rate"] = np.asarray(
            cost / float(self._cfg.self_collision_history_length), dtype=np.float32
        )
        return cost

    def _compute_reward(self, info, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        reward = super()._compute_reward(info, linvel, gyro, gravity, dof_pos, dof_vel)
        steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
        if self._enable_reward_log and int(steps[0]) % 4 == 0:
            log = info.setdefault("log", {})
            log["reward/self_collision_hit_count_mean"] = float(
                np.mean(info["self_collision_hit_count"])
            )
            log["reward/self_collision_hit_rate_mean"] = float(
                np.mean(info["self_collision_hit_rate"])
            )
        return reward

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        base = super().obs_groups_spec
        return {
            "obs": base["obs"] - 2,
            "critic": base["critic"] - 2,
            "amp": AMP_OBSERVATION_DIM,
        }

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        obs = G1WalkEnv._compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel)
        return self._add_amp_observation(obs)

    def _compute_obs_for_rows(
        self,
        info: dict,
        linvel,
        gyro,
        gravity,
        dof_pos,
        dof_vel,
        *,
        env_ids,
    ) -> dict[str, np.ndarray]:
        obs = G1WalkEnv._compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel)
        return self._add_amp_observation(obs, env_ids=np.asarray(env_ids, dtype=np.intp))

    def _add_amp_observation(
        self, obs: dict[str, np.ndarray], *, env_ids: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        phase_start = 3 + 3 + 3 * self._num_action + 3
        obs["obs"] = np.concatenate(
            (obs["obs"][:, :phase_start], obs["obs"][:, phase_start + 2 :]), axis=1
        )
        obs["critic"] = np.concatenate(
            (obs["critic"][:, :phase_start], obs["critic"][:, phase_start + 2 :]), axis=1
        )
        obs["amp"] = self._compute_amp_observation(env_ids)
        return obs

    def _compute_amp_observation(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        pos, quat, lin_vel, ang_vel = self._backend.get_body_state_w(self._amp_body_ids_with_anchor)
        if env_ids is not None:
            pos = pos[env_ids]
            quat = quat[env_ids]
            lin_vel = lin_vel[env_ids]
            ang_vel = ang_vel[env_ids]
        body_count = len(AMP_BODY_NAMES)
        return build_amp_observation_from_selected(
            body_pos_w=pos[:, :body_count],
            body_quat_w=quat[:, :body_count],
            body_lin_vel_w=lin_vel[:, :body_count],
            body_ang_vel_w=ang_vel[:, :body_count],
            anchor_pos_w=pos[:, body_count],
            anchor_quat_w=quat[:, body_count],
        )


registry.register_env("G1AMPWalk", G1AMPWalkEnv, sim_backend="mujoco")
