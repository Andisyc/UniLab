"""Shared core for interactive policy playback entrypoints."""

from __future__ import annotations

import copy
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np
import torch

LogFn = Callable[[str], None]


def _ensure_scripts_dir(root_dir: str | Path) -> None:
    scripts_dir = Path(root_dir) / "scripts"
    if scripts_dir.is_dir() and str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def _actor_input_dim_from_state_dict(state_dict: Mapping[str, Any]) -> int | None:
    for key in ("net.0.weight", "actor.net.0.weight", "mlp.0.weight", "actor.mlp.0.weight"):
        weight = state_dict.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.shape[1])
    for key, weight in state_dict.items():
        if key.endswith(".0.weight") and isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.shape[1])
    return None


_LEGACY_TAR_WEIGHTS_ONLY_ERROR = (
    "Cannot use ``weights_only=True`` with files saved in the legacy .tar format"
)


def _load_playback_checkpoint(checkpoint_path: str, *, device_name: str, log: LogFn) -> Any:
    try:
        return torch.load(checkpoint_path, map_location=device_name, weights_only=True)
    except RuntimeError as exc:
        if _LEGACY_TAR_WEIGHTS_ONLY_ERROR not in str(exc):
            raise
        log(
            "WARNING: checkpoint uses legacy PyTorch .tar serialization; "
            "reloading with weights_only=False. Only use trusted local checkpoints."
        )
        try:
            return torch.load(checkpoint_path, map_location=device_name, weights_only=False)
        except Exception as legacy_exc:
            raise RuntimeError(
                "Failed to load checkpoint after PyTorch legacy .tar fallback: "
                f"{checkpoint_path}. The file may be corrupted, incomplete, or not a "
                "PyTorch checkpoint; re-copy or re-download the checkpoint before playback."
            ) from legacy_exc


@dataclass(frozen=True)
class RslRlPlaybackConfig:
    """Configuration needed to bootstrap an RSL-RL interactive playback session."""

    task: str
    load_run: str
    checkpoint: str | None
    action_mode: str
    policy_obs_mode: str
    algo_log_name: str
    log_root: str | None
    num_envs: int = 1
    speed: float = 1.0
    start_paused: bool = False
    checkpoint_path: str | None = None


@dataclass
class PlaybackControls:
    """Viewer-independent playback control state."""

    paused: bool = False
    speed: float = 1.0
    _single_step_requests: int = field(default=0, init=False, repr=False)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def request_single_step(self, count: int = 1) -> None:
        self._single_step_requests += max(int(count), 0)

    def set_speed(self, value: float) -> None:
        self.speed = max(float(value), 1e-6)

    def consume_step_permission(self) -> bool:
        if self.paused:
            if self._single_step_requests <= 0:
                return False
            self._single_step_requests -= 1
            return True
        if self._single_step_requests > 0:
            self._single_step_requests -= 1
        return True

    def target_dt(self, ctrl_dt: float) -> float:
        return float(ctrl_dt) / max(float(self.speed), 1e-6)


@dataclass
class KeyboardCommander:
    """Mutable ``[vx, vy, vyaw]`` velocity command driven by keyboard nudges.

    Per-axis nudges stack and are clamped to the task's ``commands.vel_limit``.
    """

    low: np.ndarray
    high: np.ndarray
    step_lin: float = 0.1
    step_ang: float = 0.2
    command: np.ndarray = field(init=False)

    AXIS_VX: ClassVar[int] = 0
    AXIS_VY: ClassVar[int] = 1
    AXIS_VYAW: ClassVar[int] = 2

    def __post_init__(self) -> None:
        self.low = np.asarray(self.low, dtype=np.float64).reshape(3)
        self.high = np.asarray(self.high, dtype=np.float64).reshape(3)
        self.command = np.zeros(3, dtype=np.float64)

    @classmethod
    def from_vel_limit(
        cls, vel_limit: Any, *, step_lin: float = 0.1, step_ang: float = 0.2
    ) -> "KeyboardCommander":
        limit = np.asarray(vel_limit, dtype=np.float64)
        if limit.shape != (2, 3):
            raise ValueError(f"commands.vel_limit must have shape (2, 3), got {limit.shape}")
        return cls(low=limit[0], high=limit[1], step_lin=float(step_lin), step_ang=float(step_ang))

    def nudge(self, axis: int, sign: float) -> None:
        base = self.step_lin if axis in (self.AXIS_VX, self.AXIS_VY) else self.step_ang
        delta = base * (1.0 if sign >= 0 else -1.0)
        self.command[axis] = float(
            np.clip(self.command[axis] + delta, self.low[axis], self.high[axis])
        )

    def zero(self) -> None:
        self.command[:] = 0.0

    def describe(self) -> str:
        return (
            f"cmd vx={self.command[0]:+.2f} vy={self.command[1]:+.2f} vyaw={self.command[2]:+.2f}"
        )


@dataclass(frozen=True)
class MotionOverlaySelection:
    """Cold-path selection of task bodies used by playback overlays."""

    enabled: bool
    selected_indices: np.ndarray


class PlaybackSession(Protocol):
    """Viewer-facing session contract shared by all policy families."""

    env: Any

    def reset(self) -> Any: ...

    def refresh_observation(self) -> Any: ...

    def advance(self, controls: PlaybackControls) -> bool: ...

    def physics_state(self) -> np.ndarray: ...

    @property
    def info(self) -> dict[str, Any]: ...


class RslRlPlaybackSession:
    """Policy/action stepping core shared by native and web viewers."""

    def __init__(
        self,
        *,
        env: Any,
        wrapped_env: Any,
        device: str,
        action_mode: str,
        policy: Callable[[Any], Any] | None,
        num_envs: int,
    ) -> None:
        self.env = env
        self.wrapped_env = wrapped_env
        self.device = device
        self.action_mode = action_mode
        self.policy = policy
        self.num_envs = int(num_envs)
        self.obs: Any | None = None
        self.action_obs: Any | None = None
        self.actions: torch.Tensor | None = None
        self.step_count = 0

    def reset(self) -> Any:
        self.obs, _info = self.wrapped_env.reset()
        self.action_obs = None
        self.actions = None
        self.step_count = 0
        return self.obs

    def refresh_observation(self) -> Any:
        """Reload the current env observation without advancing the session."""

        get_observations = getattr(self.wrapped_env, "get_observations", None)
        if not callable(get_observations):
            raise RuntimeError(
                "Playback observation refresh requires wrapped_env.get_observations()."
            )
        self.obs = get_observations()
        return self.obs

    def set_external_command(self, command: np.ndarray) -> Any:
        """Apply a velocity command and refresh every policy-facing observation."""

        state = getattr(self.env, "state", None)
        info = getattr(state, "info", None)
        commands = info.get("commands") if isinstance(info, dict) else None
        if not isinstance(commands, np.ndarray) or commands.ndim != 2 or commands.shape[1] < 3:
            raise RuntimeError(
                "Playback command synchronization requires env.state.info['commands'] "
                "with shape (num_envs, >=3)."
            )
        command_arr = np.asarray(command, dtype=commands.dtype)
        if command_arr.shape == (3,):
            command_arr = np.broadcast_to(command_arr, (commands.shape[0], 3))
        if command_arr.shape != (commands.shape[0], 3):
            raise ValueError(
                "Playback command synchronization expects command shape "
                f"(3,) or ({commands.shape[0]}, 3), got {command_arr.shape}."
            )
        if np.array_equal(commands[:, :3], command_arr):
            return self.obs

        commands[:, :3] = command_arr
        refresh_state = getattr(self.env, "refresh_state", None)
        if not callable(refresh_state):
            raise RuntimeError("Playback command synchronization requires env.refresh_state().")
        refresh_state()
        return self.refresh_observation()

    def step_once(self) -> Any:
        actions = self._build_actions()
        self.actions = actions
        self.obs, _reward, _done, _info = self.wrapped_env.step(actions)
        self.step_count += 1
        return self.obs

    def advance(self, controls: PlaybackControls) -> bool:
        if not controls.consume_step_permission():
            return False
        self.step_once()
        return True

    def physics_state(self) -> np.ndarray:
        return self.env.get_physics_state_snapshot()

    @property
    def info(self) -> dict[str, Any]:
        state = getattr(self.env, "state", None)
        info = getattr(state, "info", None)
        return info if isinstance(info, dict) else {}

    def _build_actions(self) -> torch.Tensor:
        if self.obs is None:
            raise RuntimeError("Playback session must be reset before stepping.")
        self.action_obs = self.obs
        action_space = self.env.action_space
        action_dim = int(action_space.shape[0])
        if self.action_mode == "policy" and self.policy is not None:
            return self.policy(self.obs)
        if self.action_mode == "random":
            actions = np.random.uniform(
                action_space.low,
                action_space.high,
                size=(self.num_envs, action_dim),
            )
            return torch.from_numpy(actions).to(self.device).float()
        return torch.zeros(self.num_envs, action_dim, device=self.device)


class OffPolicyPlaybackSession:
    """Direct env stepping session for SAC-style off-policy actors."""

    def __init__(
        self,
        *,
        env: Any,
        device: str,
        action_mode: str,
        actor: Any | None,
        actor_algo_type: str,
        normalizer: Any | None,
        num_envs: int,
        obs_extractor: Callable[[dict[str, np.ndarray]], np.ndarray],
        priv_info_resolver: Callable[..., np.ndarray | None],
    ) -> None:
        self.env = env
        self.device = device
        self.action_mode = action_mode
        self.actor = actor
        self.actor_algo_type = str(actor_algo_type)
        self.normalizer = normalizer
        self.num_envs = int(num_envs)
        self.obs_extractor = obs_extractor
        self.priv_info_resolver = priv_info_resolver
        self.obs: np.ndarray | None = None
        self.current_priv_info: np.ndarray | None = None
        self.step_count = 0

    def reset(self) -> np.ndarray:
        if self.env.state is None:
            self.env.init_state()
        env_indices = np.arange(self.num_envs, dtype=np.int32)
        reset_result = self.env.reset(env_indices)
        if not isinstance(reset_result, tuple) or len(reset_result) != 2:
            raise ValueError(f"Unexpected env.reset return format: {type(reset_result)!r}")
        obs_out, info_out = reset_result
        self.obs = np.asarray(self.obs_extractor(obs_out), dtype=np.float32)
        self.current_priv_info = self._resolve_priv_info(obs_out, info_out)
        self.step_count = 0
        return self.obs

    def refresh_observation(self) -> np.ndarray:
        """Reload the current env observation without advancing the session."""

        state = getattr(self.env, "state", None)
        obs_out = getattr(state, "obs", None)
        if not isinstance(obs_out, dict):
            raise RuntimeError(
                "Off-policy playback observation refresh requires env.state.obs as a dict."
            )
        info_out = getattr(state, "info", None)
        self.obs = np.asarray(self.obs_extractor(obs_out), dtype=np.float32)
        self.current_priv_info = self._resolve_priv_info(
            obs_out,
            info_out if isinstance(info_out, dict) else None,
        )
        return self.obs

    def step_once(self) -> np.ndarray:
        actions = self._build_actions()
        state = self.env.step(actions)
        self.obs = np.asarray(self.obs_extractor(state.obs), dtype=np.float32)
        self.current_priv_info = self._resolve_priv_info(state.obs, state.info)
        self.step_count += 1
        return self.obs

    def advance(self, controls: PlaybackControls) -> bool:
        if not controls.consume_step_permission():
            return False
        self.step_once()
        return True

    def physics_state(self) -> np.ndarray:
        return self.env.get_physics_state_snapshot()

    @property
    def info(self) -> dict[str, Any]:
        state = getattr(self.env, "state", None)
        info = getattr(state, "info", None)
        return info if isinstance(info, dict) else {}

    def _resolve_priv_info(
        self,
        obs_dict: dict[str, np.ndarray],
        info: dict[str, Any] | None,
    ) -> np.ndarray | None:
        if self.actor_algo_type != "hora_sac":
            return None
        if self.action_mode != "policy" or self.actor is None:
            return None
        from unilab.base.observations import split_obs_dict

        actor_obs_np, critic_np = split_obs_dict(obs_dict)
        priv_info = self.priv_info_resolver(
            algo_type=self.actor_algo_type,
            obs_np=np.asarray(actor_obs_np, dtype=np.float32),
            critic_np=np.asarray(critic_np, dtype=np.float32),
            info=info,
        )
        if priv_info is None:
            raise ValueError("HORA-SAC interactive play step is missing privileged info.")
        return np.asarray(priv_info, dtype=np.float32)

    def _build_actions(self) -> np.ndarray:
        if self.obs is None:
            raise RuntimeError("Playback session must be reset before stepping.")
        action_space = self.env.action_space
        action_dim = int(action_space.shape[0])
        if self.action_mode == "policy" and self.actor is not None:
            obs_torch = torch.from_numpy(self.obs).to(self.device)
            if self.normalizer is not None:
                obs_torch = self.normalizer(obs_torch, update=False)
            if self.actor_algo_type == "hora_sac":
                if self.current_priv_info is None:
                    raise ValueError("HORA-SAC interactive play step is missing privileged info.")
                priv_info_torch = torch.from_numpy(self.current_priv_info).to(self.device)
                actions = self.actor.explore(
                    obs_torch,
                    priv_info_torch,
                    deterministic=True,
                )
            else:
                actions = self.actor.explore(obs_torch, deterministic=True)
            return actions.detach().cpu().numpy().astype(np.float32)
        if self.action_mode == "random":
            return np.random.uniform(
                action_space.low,
                action_space.high,
                size=(self.num_envs, action_dim),
            ).astype(np.float32)
        return np.zeros((self.num_envs, action_dim), dtype=np.float32)


_HORA_DISTILL_CHECKPOINT_UNAVAILABLE = "hora_distill_checkpoint_unavailable"


def select_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_rsl_rl_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    env_factory: Callable[[int], Any],
    algo_config: dict[str, Any],
    root_dir: str | Path,
    device: str | None,
    checkpoint_resolver: Callable[[str, str, str | None, str, str | None], str | None],
    checkpoint_input_dim_reader: Callable[[str], int | None],
    entrypoint_log_root: Callable[..., Path],
    wrapper_cls: Any,
    runner_cls: Any,
    policy_obs_dims_getter: Callable[[Any], tuple[int, int]],
    train_cfg_normalizer: Callable[[dict[str, Any]], dict[str, Any]],
    log: LogFn = print,
) -> tuple[RslRlPlaybackSession, str, str | None]:
    """Create a playback session and load the selected policy checkpoint."""

    device_name = select_torch_device() if device is None else str(device)
    env = env_factory(int(playback_cfg.num_envs))
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")
    actor_obs_dim, flat_obs_dim = policy_obs_dims_getter(env.obs_groups_spec)

    policy_obs_mode = playback_cfg.policy_obs_mode
    checkpoint_path: str | None = None
    if playback_cfg.action_mode == "policy":
        checkpoint_path = checkpoint_resolver(
            playback_cfg.task,
            playback_cfg.load_run,
            playback_cfg.checkpoint,
            playback_cfg.algo_log_name,
            playback_cfg.log_root,
        )
        if policy_obs_mode == "auto" and checkpoint_path is not None:
            ckpt_dim = checkpoint_input_dim_reader(checkpoint_path)
            if ckpt_dim == actor_obs_dim:
                policy_obs_mode = "actor"
            elif ckpt_dim == flat_obs_dim:
                policy_obs_mode = "flat"
            elif ckpt_dim is not None:
                raise RuntimeError(
                    "Checkpoint actor input dim mismatch: "
                    f"ckpt={ckpt_dim}, actor_obs={actor_obs_dim}, flat_obs={flat_obs_dim}. "
                    "Please pass --policy_obs_mode actor|flat explicitly if needed."
                )
            else:
                policy_obs_mode = "flat"

    wrapped_env = wrapper_cls(env, device=device_name, policy_obs_mode=policy_obs_mode)
    log(f"Policy obs mode: {policy_obs_mode} (actor_obs={actor_obs_dim}, flat_obs={flat_obs_dim})")

    train_cfg = train_cfg_normalizer(copy.deepcopy(algo_config))
    if "runner" not in train_cfg:
        train_cfg["runner"] = {}
    train_cfg["runner"]["logger"] = "none"

    policy = None
    if playback_cfg.action_mode == "policy":
        if checkpoint_path is None:
            log("WARNING: no checkpoint found - falling back to zero actions.")
        else:
            log_dir = str(
                entrypoint_log_root(
                    Path(root_dir),
                    algo_log_name=playback_cfg.algo_log_name,
                    log_root=playback_cfg.log_root,
                )
                / playback_cfg.task
                / "play_temp"
            )
            runner = runner_cls(wrapped_env, train_cfg, log_dir=log_dir, device=device_name)
            runner.load(
                checkpoint_path,
                load_cfg={
                    "actor": True,
                    "critic": False,
                    "optimizer": False,
                    "iteration": False,
                    "rnd": False,
                },
            )
            policy = runner.get_inference_policy(device=device_name)

    log(f"Action mode: {playback_cfg.action_mode}")
    session = RslRlPlaybackSession(
        env=env,
        wrapped_env=wrapped_env,
        device=device_name,
        action_mode=playback_cfg.action_mode,
        policy=policy,
        num_envs=playback_cfg.num_envs,
    )
    return session, policy_obs_mode, checkpoint_path


def _normalize_checkpoint_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text in {"", "-1", "None", "null"} else text


def _cfg_checkpoint_value(cfg: Any) -> str | None:
    from omegaconf import OmegaConf

    return _normalize_checkpoint_value(OmegaConf.select(cfg, "algo.checkpoint", default=None))


def _resolve_appo_checkpoint_from_cfg(
    cfg: Any,
    *,
    root_dir: str | Path,
) -> tuple[str | None, str | None]:
    _ensure_scripts_dir(root_dir)
    from unilab.training import get_log_root, resolve_task_checkpoint_path

    selected_checkpoint = _cfg_checkpoint_value(cfg)
    if selected_checkpoint is not None:
        checkpoint_path, checkpoint_dir = resolve_task_checkpoint_path(
            root_dir,
            task_name=str(cfg.training.task_name),
            load_run=str(cfg.algo.load_run),
            algo_log_name=str(cfg.algo.algo_log_name),
            checkpoint=selected_checkpoint,
            log_root=getattr(cfg.training, "log_root", None),
        )
        return (
            str(checkpoint_path) if checkpoint_path is not None else None,
            str(checkpoint_dir) if checkpoint_dir is not None else None,
        )

    from train_appo import resolve_appo_checkpoint_path

    base_log_dir = get_log_root(root_dir, cfg) / str(cfg.training.task_name)
    checkpoint_path, checkpoint_dir = resolve_appo_checkpoint_path(base_log_dir, cfg.algo.load_run)
    return (
        str(checkpoint_path) if checkpoint_path is not None else None,
        str(checkpoint_dir) if checkpoint_dir is not None else None,
    )


def _build_appo_actor(
    *,
    env: Any,
    wrapped_env: Any,
    cfg: Any,
    rl_cfg: dict[str, Any],
    device: str,
    is_hora: bool,
) -> Any:
    from copy import deepcopy

    from rsl_rl.utils import resolve_callable
    from tensordict import TensorDict

    from unilab.base.observations import get_obs_dims

    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])
    rl_cfg_dict = deepcopy(rl_cfg)

    if is_hora:
        from unilab.algos.torch.hora.appo import _update_hora_obs_groups
        from unilab.algos.torch.hora.models import build_hora_shared_actor_critic
        from unilab.algos.torch.hora.rsl_rl_compat import (
            convert_config_v3_to_v4,
            is_rsl_rl_v4,
            is_rsl_rl_v5,
        )

        obs_td = wrapped_env.get_observations()
        num_envs = int(getattr(wrapped_env, "num_envs", getattr(env, "num_envs", 1)))
        obs_dim = int(obs_td["actor"].shape[-1])
        priv_info_dim = int(obs_td["priv_info"].shape[-1])
        if priv_info_dim <= 0:
            raise ValueError("HORA APPO interactive play requires privileged info.")
        _update_hora_obs_groups(rl_cfg_dict, obs_dim=obs_dim, priv_info_dim=priv_info_dim)
        if is_rsl_rl_v5():
            pass
        elif is_rsl_rl_v4():
            rl_cfg_dict = convert_config_v3_to_v4(rl_cfg_dict)

        actor_cfg = deepcopy(rl_cfg_dict["actor"])
        actor_cls = resolve_callable(actor_cfg.pop("class_name"))
        actor_cfg.pop("num_actions", None)
        critic_cfg = deepcopy(rl_cfg_dict.get("critic") or rl_cfg_dict.get("actor") or {})
        critic_cfg.pop("class_name", None)
        critic_cfg.pop("num_actions", None)
        critic_cfg.pop("distribution_cfg", None)
        shared_model = build_hora_shared_actor_critic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            priv_info_dim=priv_info_dim,
            actor_cfg=actor_cfg,
            critic_cfg=critic_cfg,
        ).to(device)
        td_example = TensorDict(
            {
                "actor": torch.zeros((num_envs, obs_dim), device=device),
                "priv_info": torch.zeros(
                    (num_envs, priv_info_dim),
                    device=device,
                ),
            },
            batch_size=num_envs,
        )
        actor = actor_cls(
            td_example,
            rl_cfg_dict["obs_groups"],
            "actor",
            action_dim,
            shared_model=shared_model,
            **actor_cfg,
        )
        return actor.to(device).eval()

    obs_dim, critic_dim = get_obs_dims(env.obs_groups_spec)
    num_envs = int(getattr(wrapped_env, "num_envs", getattr(env, "num_envs", 1)))
    obs_groups = rl_cfg_dict.setdefault("obs_groups", {})
    if "obs_groups" not in rl_cfg_dict or not isinstance(obs_groups, dict):
        obs_groups = {}
        rl_cfg_dict["obs_groups"] = obs_groups
    actor_group = obs_groups.get("actor", obs_groups.get("policy", {}))
    if isinstance(actor_group, dict) and "policy" in actor_group:
        actor_group["policy"] = obs_dim
        obs_groups["actor"] = actor_group
    else:
        obs_groups["actor"] = {"policy": obs_dim}
    critic_group = obs_groups.get("critic")
    if critic_group is None:
        obs_groups["critic"] = {"policy": critic_dim if critic_dim > 0 else obs_dim}
    elif isinstance(critic_group, dict) and "policy" in critic_group:
        critic_group["policy"] = critic_dim if critic_dim > 0 else obs_dim

    obs_example = torch.zeros((num_envs, obs_dim), device=device)
    td_example = TensorDict({"policy": obs_example}, batch_size=num_envs)
    actor_cfg = deepcopy(rl_cfg_dict["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor_cfg.pop("num_actions", None)
    actor = actor_cls(td_example, rl_cfg_dict["obs_groups"], "actor", action_dim, **actor_cfg)
    return actor.to(device).eval()


def create_appo_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    rl_cfg: dict[str, Any],
    env_factory: Callable[[int], Any],
    root_dir: str | Path,
    device: str | None,
    wrapper_cls: Any,
    log: LogFn = print,
) -> tuple[RslRlPlaybackSession, str, str | None]:
    """Create an APPO interactive playback session."""

    device_name = select_torch_device() if device is None else str(device)
    env = env_factory(int(playback_cfg.num_envs))
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    from unilab.algos.torch.hora.runtime import is_hora_appo_runtime

    is_hora = is_hora_appo_runtime(rl_cfg)
    selected_wrapper_cls = wrapper_cls
    policy_obs_mode = playback_cfg.policy_obs_mode
    if is_hora:
        from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper

        selected_wrapper_cls = HoraRslRlVecEnvWrapper
        policy_obs_mode = "actor"

    wrapped_env = selected_wrapper_cls(env, device=device_name, policy_obs_mode=policy_obs_mode)
    policy = None
    checkpoint_path: str | None = None
    if playback_cfg.action_mode == "policy":
        checkpoint_path, _checkpoint_dir = _resolve_appo_checkpoint_from_cfg(cfg, root_dir=root_dir)
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            log(
                "WARNING: no APPO checkpoint found for "
                f"load_run={cfg.algo.load_run} - falling back to zero actions."
            )
        else:
            actor = _build_appo_actor(
                env=env,
                wrapped_env=wrapped_env,
                cfg=cfg,
                rl_cfg=rl_cfg,
                device=device_name,
                is_hora=is_hora,
            )
            checkpoint = _load_playback_checkpoint(
                checkpoint_path,
                device_name=device_name,
                log=log,
            )
            actor.load_state_dict(checkpoint["actor"])
            policy = actor
            log(f"Loading APPO checkpoint: {checkpoint_path}")

    log(f"Action mode: {playback_cfg.action_mode}")
    return (
        RslRlPlaybackSession(
            env=env,
            wrapped_env=wrapped_env,
            device=device_name,
            action_mode=playback_cfg.action_mode,
            policy=policy,
            num_envs=playback_cfg.num_envs,
        ),
        policy_obs_mode,
        checkpoint_path,
    )


def create_sac_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    env_factory: Callable[[int], Any],
    root_dir: str | Path,
    device: str | None,
    algo_name: str = "sac",
    log: LogFn = print,
) -> tuple[OffPolicyPlaybackSession, str, str | None]:
    """Create an interactive playback session for off-policy actors."""

    import os

    _ensure_scripts_dir(root_dir)

    from train_offpolicy import (
        default_device,
        extract_play_obs,
        resolve_checkpoint_path,
        resolve_play_actor_spec,
        resolve_play_obs_dims,
    )

    from unilab.algos.torch.common.actor_factory import build_actor
    from unilab.algos.torch.offpolicy.worker import resolve_offpolicy_actor_priv_info

    device_name = default_device(torch, str(device) if device is not None else None)
    env = env_factory(int(playback_cfg.num_envs))
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    obs_dim, critic_obs_dim = resolve_play_obs_dims(env.obs_groups_spec)
    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])
    actor_algo_type, actor_kwargs = resolve_play_actor_spec(
        algo_name,
        cfg,
        obs_dim=obs_dim,
        critic_obs_dim=critic_obs_dim,
    )
    if algo_name == "flashsac":
        actor_kwargs.update(
            {
                "actor_num_blocks": cfg.algo.algo_params.actor_num_blocks,
                "actor_noise_zeta_mu": cfg.algo.algo_params.actor_noise_zeta_mu,
                "actor_noise_zeta_max": cfg.algo.algo_params.actor_noise_zeta_max,
            }
        )

    actor = None
    checkpoint_path: str | None = None
    normalizer = None
    if bool(getattr(cfg.algo, "obs_normalization", False)):
        from unilab.algos.torch.common.normalization import EmpiricalNormalization

        normalizer = EmpiricalNormalization(shape=obs_dim, device=device_name)
    if playback_cfg.action_mode == "policy":
        actor = build_actor(
            actor_algo_type,
            obs_dim,
            action_dim,
            cfg.algo.actor_hidden_dim,
            cfg.algo.use_layer_norm,
            device_name,
            **actor_kwargs,
        )
        actor.eval()
        checkpoint_path, _checkpoint_dir = resolve_checkpoint_path(
            Path(root_dir),
            cfg.algo.algo_log_name,
            cfg.training.task_name,
            cfg.algo.load_run,
        )
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            log(
                f"WARNING: no {algo_name} checkpoint found for "
                f"load_run={cfg.algo.load_run} - falling back to zero actions."
            )
            actor = None
        else:
            checkpoint = _load_playback_checkpoint(
                checkpoint_path,
                device_name=device_name,
                log=log,
            )
            checkpoint_actor = checkpoint["actor"]
            checkpoint_obs_dim = _actor_input_dim_from_state_dict(checkpoint_actor)
            if checkpoint_obs_dim is not None and checkpoint_obs_dim != obs_dim:
                raise RuntimeError(
                    "Off-policy checkpoint actor input dim mismatch: "
                    f"checkpoint={checkpoint_obs_dim}, playback_env_obs={obs_dim}. "
                    "The playback env contract does not match the selected run_config. "
                    "For G1 mode-conditioned policies, ensure env.mode_observation is restored "
                    "from the checkpoint run_config or pass the matching Hydra overrides."
                )
            actor.load_state_dict(checkpoint_actor)
            if normalizer is not None and checkpoint.get("obs_normalizer"):
                normalizer.load_state_dict(checkpoint["obs_normalizer"])
                normalizer.eval()
            log(f"Loading {algo_name} checkpoint: {checkpoint_path}")

    log(f"Action mode: {playback_cfg.action_mode}")
    return (
        OffPolicyPlaybackSession(
            env=env,
            device=device_name,
            action_mode=playback_cfg.action_mode,
            actor=actor,
            actor_algo_type=actor_algo_type,
            normalizer=normalizer,
            num_envs=playback_cfg.num_envs,
            obs_extractor=extract_play_obs,
            priv_info_resolver=resolve_offpolicy_actor_priv_info,
        ),
        "actor",
        checkpoint_path,
    )


def _default_hora_distill_playback_deps(root_dir: str | Path) -> dict[str, Any]:
    _ensure_scripts_dir(root_dir)
    from train_hora_distill import (
        _apply_teacher_defaults,
        _build_play_env_cfg_override,
        _cfg_with_checkpoint_runtime,
        _format_stage2_play_checkpoint_error,
        _resolve_stage2_checkpoint_path,
        _student_policy,
    )

    from unilab.algos.torch.hora.distill import (
        build_student_actor_and_normalizer,
        load_distilled_checkpoint,
    )
    from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper
    from unilab.training import create_env, get_log_root

    return {
        "apply_teacher_defaults": _apply_teacher_defaults,
        "build_play_env_cfg_override": _build_play_env_cfg_override,
        "build_student_actor_and_normalizer": build_student_actor_and_normalizer,
        "cfg_with_checkpoint_runtime": _cfg_with_checkpoint_runtime,
        "create_env": create_env,
        "format_stage2_play_checkpoint_error": _format_stage2_play_checkpoint_error,
        "get_log_root": get_log_root,
        "load_distilled_checkpoint": load_distilled_checkpoint,
        "resolve_stage2_checkpoint_path": _resolve_stage2_checkpoint_path,
        "student_policy": _student_policy,
        "wrapper_cls": HoraRslRlVecEnvWrapper,
        "checkpoint_reader": torch.load,
    }


def create_hora_distill_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    root_dir: str | Path,
    device: str | None,
    deps: Mapping[str, Any] | None = None,
    log: LogFn = print,
) -> tuple[RslRlPlaybackSession, str, str | None]:
    """Create an interactive playback session for HORA stage-2 student checkpoints."""

    resolved_deps = dict(_default_hora_distill_playback_deps(root_dir) if deps is None else deps)
    device_name = select_torch_device() if device is None else str(device)
    load_path, load_path_dir = resolved_deps["resolve_stage2_checkpoint_path"](cfg)
    checkpoint_path = str(load_path) if load_path is not None else None
    policy: Callable[[Any], Any] | None = None

    if playback_cfg.action_mode == "policy":
        if load_path is None or load_path_dir is None or not Path(load_path).exists():
            task_log_root = resolved_deps["get_log_root"](Path(root_dir), cfg) / str(
                cfg.training.task_name
            )
            log(
                resolved_deps["format_stage2_play_checkpoint_error"](
                    cfg,
                    task_log_root=task_log_root,
                    load_path=load_path,
                    load_path_dir=load_path_dir,
                )
            )
            log("WARNING: falling back to zero actions.")
            runtime_cfg = resolved_deps["apply_teacher_defaults"](cfg)
        else:
            log(f"Loading distilled checkpoint: {load_path}")
            checkpoint = resolved_deps["checkpoint_reader"](
                load_path, map_location="cpu", weights_only=False
            )
            if "model_state_dict" not in checkpoint:
                raise ValueError(
                    f"Checkpoint at {load_path} is not a HORA distillation checkpoint "
                    f"(found keys: {set(checkpoint.keys())})."
                )
            runtime_cfg = resolved_deps["cfg_with_checkpoint_runtime"](cfg, checkpoint)
    else:
        runtime_cfg = resolved_deps["apply_teacher_defaults"](cfg)

    env_cfg_override = resolved_deps["build_play_env_cfg_override"](runtime_cfg)
    create_env = resolved_deps["create_env"]
    try:
        env = create_env(
            runtime_cfg,
            num_envs=int(playback_cfg.num_envs),
            env_cfg_override=env_cfg_override,
            sim_backend="mujoco",
            task_name=str(runtime_cfg.training.task_name),
        )
    except TypeError:
        if deps is None:
            raise
        env = create_env(
            runtime_cfg,
            num_envs=int(playback_cfg.num_envs),
            env_cfg_override=env_cfg_override,
        )
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    policy_obs_mode = "actor"
    wrapper_cls = resolved_deps["wrapper_cls"]
    wrapped_env = wrapper_cls(env, device=device_name, policy_obs_mode=policy_obs_mode)
    torch_device = torch.device(device_name)

    if playback_cfg.action_mode == "policy" and load_path is not None and Path(load_path).exists():
        actor, hist_normalizer = resolved_deps["build_student_actor_and_normalizer"](
            wrapped_env,
            runtime_cfg,
            device=torch_device,
        )
        resolved_deps["load_distilled_checkpoint"](
            actor,
            hist_normalizer,
            load_path,
            device=torch_device,
        )
        actor.eval()
        hist_normalizer.eval()
        student_policy = resolved_deps["student_policy"]

        def policy(obs: Any) -> Any:
            return student_policy(actor, hist_normalizer, obs, device=torch_device)

    log(f"Policy obs mode: {policy_obs_mode}")
    log(f"Action mode: {playback_cfg.action_mode}")
    session = RslRlPlaybackSession(
        env=env,
        wrapped_env=wrapped_env,
        device=device_name,
        action_mode=playback_cfg.action_mode,
        policy=policy,
        num_envs=playback_cfg.num_envs,
    )
    return session, policy_obs_mode, checkpoint_path


def _resolve_distill_checkpoint_from_playback_cfg(
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    root_dir: str | Path,
) -> Path | None:
    from unilab.training.run import resolve_task_checkpoint_path

    if playback_cfg.checkpoint_path not in (None, ""):
        path = Path(str(playback_cfg.checkpoint_path))
        resolved_path = path if path.is_absolute() else Path(root_dir) / path
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"training.play_checkpoint_path does not exist: {resolved_path}"
            )
        return resolved_path

    checkpoint_path, _run_dir = resolve_task_checkpoint_path(
        root_dir,
        task_name=str(getattr(cfg.training, "task_name", playback_cfg.task)),
        load_run=playback_cfg.load_run,
        algo_log_name=playback_cfg.algo_log_name,
        checkpoint=playback_cfg.checkpoint,
        log_root=playback_cfg.log_root,
    )
    return checkpoint_path


def _apply_distill_playback_reset_contract(
    env_cfg_override: Mapping[str, Any] | None, task_name: str
) -> dict[str, Any] | None:
    """Force standing-only reset sampling for G1 distill playback owners."""

    task_key = str(task_name).lower().split("/", 1)[0].replace("-", "_")
    task_key = task_key.replace("_", "")
    if task_key not in {"g1walkflat", "g1walkheight"}:
        return dict(env_cfg_override) if env_cfg_override is not None else None
    merged = dict(env_cfg_override or {})
    commands_override = dict(merged.get("commands") or {})
    commands_override["rel_standing_envs"] = 1.0
    if "rel_transition_envs" in commands_override:
        commands_override["rel_transition_envs"] = 0.0
    merged["commands"] = commands_override
    if "standing_reset_base_qvel_limit" in merged:
        merged["standing_reset_base_qvel_limit"] = 0.0
    return merged


def _default_distill_playback_deps(root_dir: str | Path) -> dict[str, Any]:
    _ensure_scripts_dir(root_dir)
    from unilab.algos.torch.distill import load_distillation_student_policy
    from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper
    from unilab.training import BackendAdapter, create_env, ensure_registries

    ensure_registries()

    return {
        "build_env_cfg_override": lambda cfg: BackendAdapter(
            cfg,
            root_dir=root_dir,
            algo_name="distill",
        ).build_task_env_cfg_override(),
        "create_env": create_env,
        "load_student_policy": load_distillation_student_policy,
        "resolve_checkpoint": _resolve_distill_checkpoint_from_playback_cfg,
        "wrapper_cls": HoraRslRlVecEnvWrapper,
    }


def _distill_student_obs_tensor(obs: Any, *, device: str | torch.device) -> torch.Tensor:
    if isinstance(obs, Mapping):
        if "obs" in obs:
            obs = obs["obs"]
        elif "actor" in obs:
            obs = obs["actor"]
    if isinstance(obs, torch.Tensor):
        return obs.to(device=device, dtype=torch.float32)
    return torch.as_tensor(obs, dtype=torch.float32, device=device)


def distill_command_intents_from_commands(
    commands: Any,
    *,
    xy_threshold: float = 0.05,
    yaw_threshold: float = 0.05,
) -> tuple[str, ...]:
    command_array = np.asarray(commands, dtype=np.float32)
    if command_array.ndim == 1:
        command_array = command_array.reshape(1, -1)
    if command_array.ndim != 2 or command_array.shape[1] < 3:
        raise ValueError(
            "distill command intent requires commands with shape (N, >=3), "
            f"got {command_array.shape}"
        )
    if not np.isfinite(command_array[:, :3]).all():
        raise ValueError("distill command intent requires finite command values")
    xy_norm = np.linalg.norm(command_array[:, :2], axis=1)
    active = (xy_norm > float(xy_threshold)) | (np.abs(command_array[:, 2]) > float(yaw_threshold))
    return tuple("active" if bool(value) else "inactive" for value in active)


def _cfg_select(cfg: Any, dotted_path: str, default: Any = None) -> Any:
    current = cfg
    for key in dotted_path.split("."):
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
        else:
            if not hasattr(current, key):
                return default
            current = getattr(current, key)
    return current


def _distill_commands_from_env(env: Any, *, batch_size: int) -> np.ndarray | None:
    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    if not isinstance(info, Mapping) or "commands" not in info:
        return None
    commands = np.asarray(info["commands"], dtype=np.float32)
    if commands.ndim == 1:
        commands = commands.reshape(1, -1)
    if commands.ndim != 2 or commands.shape[1] < 3:
        raise ValueError(
            "distill command routing requires env.state.info['commands'] with "
            f"shape (N, >=3), got {commands.shape}"
        )
    if commands.shape[0] == 1 and int(batch_size) > 1:
        commands = np.repeat(commands, int(batch_size), axis=0)
    if commands.shape[0] != int(batch_size):
        raise ValueError(
            "distill command routing command batch mismatch: "
            f"commands={commands.shape[0]} obs_batch={int(batch_size)}"
        )
    return commands[:, :3]


def _distill_command_intent_targets(
    cfg: Any,
    runtime_cfg: Mapping[str, Any],
) -> dict[str, int]:
    targets = runtime_cfg.get("command_intent_expert_targets")
    if not isinstance(targets, Mapping):
        targets = _cfg_select(cfg, "algo.command_intent_expert_targets", None)
    if not isinstance(targets, Mapping):
        targets = {"active": 0, "inactive": 1}
    resolved = {str(key): int(value) for key, value in targets.items()}
    missing = {"active", "inactive"} - set(resolved)
    if missing:
        raise ValueError(
            f"distill command routing requires command_intent_expert_targets for {sorted(missing)}"
        )
    return resolved


def _distill_effective_command_routing_mode(
    cfg: Any,
    runtime_cfg: Mapping[str, Any],
    *,
    is_moe: bool,
) -> tuple[str, str]:
    configured = str(_cfg_select(cfg, "interactive.distill_command_routing", "auto")).lower()
    if configured not in {"none", "auto", "hard", "bias"}:
        raise ValueError(
            "interactive.distill_command_routing must be one of "
            f"none, auto, hard, bias; got {configured!r}"
        )
    if not is_moe:
        return configured, "none"
    if configured == "auto":
        coef = float(runtime_cfg.get("command_intent_loss_coef") or 0.0)
        behavior_source = str(runtime_cfg.get("expert_behavior_loss_source") or "none")
        command_intent_trained = coef > 0.0 or behavior_source == "command_intent"
        return configured, "hard" if command_intent_trained else "none"
    return configured, configured


def _distill_expected_expert_tensor(
    intents: tuple[str, ...],
    targets: Mapping[str, int],
    *,
    num_experts: int,
    device: torch.device | str,
) -> torch.Tensor:
    indices = [int(targets[intent]) for intent in intents]
    if not indices:
        return torch.empty((0,), dtype=torch.long, device=device)
    target_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
    if int(target_tensor.min().item()) < 0 or int(target_tensor.max().item()) >= int(num_experts):
        raise ValueError(
            "distill command routing expert target out of range: "
            f"targets={sorted(set(indices))} num_experts={int(num_experts)}"
        )
    return target_tensor


def create_distill_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    root_dir: str | Path,
    device: str | None,
    deps: Mapping[str, Any] | None = None,
    log: LogFn = print,
) -> tuple[RslRlPlaybackSession, str, str | None]:
    """Create a playback session for generic distillation student checkpoints."""

    resolved_deps = dict(_default_distill_playback_deps(root_dir) if deps is None else deps)
    device_name = select_torch_device() if device is None else str(device)
    checkpoint = resolved_deps["resolve_checkpoint"](playback_cfg, cfg, root_dir)
    checkpoint_path = str(checkpoint) if checkpoint is not None else None
    policy_obs_mode = playback_cfg.policy_obs_mode
    if policy_obs_mode == "auto":
        policy_obs_mode = "actor"

    create_env = resolved_deps["create_env"]
    task_name = str(getattr(cfg.training, "task_name", playback_cfg.task))
    build_env_cfg_override = resolved_deps.get("build_env_cfg_override")
    env_cfg_override = build_env_cfg_override(cfg) if build_env_cfg_override is not None else {}
    env_cfg_override = _apply_distill_playback_reset_contract(env_cfg_override, task_name)
    try:
        env = create_env(
            cfg,
            num_envs=int(playback_cfg.num_envs),
            env_cfg_override=env_cfg_override,
            sim_backend="mujoco",
            task_name=task_name,
        )
    except TypeError:
        if deps is None:
            raise
        try:
            env = create_env(
                cfg,
                num_envs=int(playback_cfg.num_envs),
                env_cfg_override=env_cfg_override,
            )
        except TypeError:
            env = create_env(cfg, num_envs=int(playback_cfg.num_envs))
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    wrapper_cls = resolved_deps["wrapper_cls"]
    wrapped_env = wrapper_cls(env, device=device_name, policy_obs_mode=policy_obs_mode)
    policy: Callable[[Any], Any] | None = None

    if playback_cfg.action_mode == "policy":
        if checkpoint is None or not Path(checkpoint).exists():
            log(
                "WARNING: no generic distillation student checkpoint found - "
                "falling back to zero actions."
            )
        else:
            log(f"Loading distillation student checkpoint: {checkpoint}")
            loaded_student = resolved_deps.get("load_student_policy")
            if loaded_student is None:
                from unilab.algos.torch.distill import load_distillation_student_policy

                loaded_student = load_distillation_student_policy
            student = loaded_student(checkpoint, device=device_name)
            raw_checkpoint = torch.load(
                Path(checkpoint),
                map_location="cpu",
                weights_only=False,
            )
            obs_normalizer_state = (
                raw_checkpoint.get("obs_normalizer")
                if isinstance(raw_checkpoint, Mapping)
                else None
            )
            obs_normalizer_keys = (
                tuple(str(key) for key in obs_normalizer_state.keys())
                if isinstance(obs_normalizer_state, Mapping)
                else ()
            )
            runtime_cfg = dict(student.distill_runtime_cfg)
            student_model_type = str(runtime_cfg.get("student_model_type", "mlp"))
            is_moe_student = student_model_type == "moe" and hasattr(student.policy, "experts")
            routing_config_mode, routing_mode = _distill_effective_command_routing_mode(
                cfg,
                runtime_cfg,
                is_moe=is_moe_student,
            )
            routing_targets = _distill_command_intent_targets(cfg, runtime_cfg)
            routing_xy_threshold = float(
                _cfg_select(cfg, "interactive.distill_command_xy_threshold", 0.05)
            )
            routing_yaw_threshold = float(
                _cfg_select(cfg, "interactive.distill_command_yaw_threshold", 0.05)
            )
            routing_bias = float(_cfg_select(cfg, "interactive.distill_command_routing_bias", 10.0))

            def policy(obs: Any) -> Any:
                obs_tensor = _distill_student_obs_tensor(obs, device=device_name)
                with torch.no_grad():
                    if not is_moe_student:
                        action = student.policy(obs_tensor).detach()
                        setattr(policy, "_unilab_distill_command_routing_applied", False)
                        setattr(policy, "_unilab_distill_last_command_intents", ())
                        setattr(policy, "_unilab_distill_last_expected_experts", ())
                        setattr(policy, "_unilab_distill_last_selected_experts", ())
                        setattr(policy, "_unilab_distill_last_route_probs", None)
                        setattr(policy, "_unilab_distill_last_raw_route_probs", None)
                        return action

                    student_output = student.policy(obs_tensor, return_diagnostics=True)
                    route_probs = student_output.route_probs
                    raw_route_probs = student_output.route_probs
                    raw_selected = torch.argmax(raw_route_probs, dim=-1)
                    selected = raw_selected
                    action = student_output.action
                    intents: tuple[str, ...] = ()
                    expected_experts: torch.Tensor | None = None
                    routing_applied = False

                    if routing_mode in {"hard", "bias"}:
                        commands = _distill_commands_from_env(
                            env,
                            batch_size=int(obs_tensor.shape[0]),
                        )
                        if commands is None:
                            raise ValueError(
                                "distill command routing requires "
                                "env.state.info['commands'] during playback"
                            )
                        intents = distill_command_intents_from_commands(
                            commands,
                            xy_threshold=routing_xy_threshold,
                            yaw_threshold=routing_yaw_threshold,
                        )
                        expected_experts = _distill_expected_expert_tensor(
                            intents,
                            routing_targets,
                            num_experts=int(student.policy.num_experts),
                            device=obs_tensor.device,
                        )
                        rows = torch.arange(
                            int(obs_tensor.shape[0]),
                            dtype=torch.long,
                            device=obs_tensor.device,
                        )
                        if routing_mode == "hard":
                            action = student_output.expert_actions[rows, expected_experts]
                            selected = expected_experts
                            route_probs = torch.nn.functional.one_hot(
                                expected_experts,
                                num_classes=int(student.policy.num_experts),
                            ).to(dtype=student_output.route_probs.dtype)
                        else:
                            biased_logits = student_output.router_logits.clone()
                            biased_logits[rows, expected_experts] += routing_bias
                            temperature = max(
                                float(getattr(student.policy, "router_temperature", 1.0)),
                                1e-8,
                            )
                            route_probs = torch.softmax(biased_logits / temperature, dim=-1)
                            selected = torch.argmax(route_probs, dim=-1)
                            action = torch.sum(
                                student_output.expert_actions * route_probs.unsqueeze(-1),
                                dim=1,
                            )
                        routing_applied = True

                    expected_tuple = (
                        tuple(int(value) for value in expected_experts.detach().cpu().tolist())
                        if expected_experts is not None
                        else ()
                    )
                    setattr(policy, "_unilab_distill_command_routing_applied", routing_applied)
                    setattr(policy, "_unilab_distill_last_command_intents", intents)
                    setattr(policy, "_unilab_distill_last_expected_experts", expected_tuple)
                    setattr(
                        policy,
                        "_unilab_distill_last_selected_experts",
                        tuple(int(value) for value in selected.detach().cpu().tolist()),
                    )
                    setattr(policy, "_unilab_distill_last_route_probs", route_probs.detach().cpu())
                    setattr(
                        policy,
                        "_unilab_distill_last_raw_route_probs",
                        raw_route_probs.detach().cpu(),
                    )
                    setattr(
                        policy,
                        "_unilab_distill_last_raw_selected_experts",
                        tuple(int(value) for value in raw_selected.detach().cpu().tolist()),
                    )
                    return action.detach()

            setattr(policy, "_unilab_distill_student_policy", student.policy)
            setattr(policy, "_unilab_distill_device", device_name)
            setattr(policy, "_unilab_distill_checkpoint_path", str(checkpoint))
            setattr(policy, "_unilab_distill_agent_steps", int(student.agent_steps))
            setattr(policy, "_unilab_distill_runtime_cfg", runtime_cfg)
            setattr(policy, "_unilab_distill_obs_normalizer_present", bool(obs_normalizer_keys))
            setattr(policy, "_unilab_distill_obs_normalizer_keys", obs_normalizer_keys)
            setattr(policy, "_unilab_distill_command_routing_mode", routing_mode)
            setattr(
                policy,
                "_unilab_distill_command_routing_config_mode",
                routing_config_mode,
            )
            setattr(policy, "_unilab_distill_command_routing_targets", dict(routing_targets))
            setattr(policy, "_unilab_distill_command_routing_applied", False)
            setattr(policy, "_unilab_distill_last_command_intents", ())
            setattr(policy, "_unilab_distill_last_expected_experts", ())
            setattr(policy, "_unilab_distill_last_selected_experts", ())
            setattr(policy, "_unilab_distill_last_route_probs", None)
            setattr(policy, "_unilab_distill_last_raw_route_probs", None)
            log(
                "Distill checkpoint diagnostics: "
                f"student_obs_dim={student.obs_dim}, "
                f"student_action_dim={student.action_dim}, "
                f"agent_steps={int(student.agent_steps)}, "
                f"obs_normalizer={'present' if obs_normalizer_keys else 'absent'}"
            )
            if routing_mode != "none":
                log(
                    "Distill command routing: "
                    f"configured={routing_config_mode}, effective={routing_mode}, "
                    f"targets={dict(routing_targets)}"
                )

    log(f"Policy obs mode: {policy_obs_mode}")
    log(f"Action mode: {playback_cfg.action_mode}")
    session = RslRlPlaybackSession(
        env=env,
        wrapped_env=wrapped_env,
        device=device_name,
        action_mode=playback_cfg.action_mode,
        policy=policy,
        num_envs=playback_cfg.num_envs,
    )
    return session, policy_obs_mode, checkpoint_path


def prepare_motion_overlay_selection(
    env: Any,
    *,
    show_target_bodies: bool,
    show_reward_debug: bool,
    target_body_names: str,
    target_max_bodies: int,
    log: LogFn = print,
) -> MotionOverlaySelection:
    """Resolve body indices used by motion-target and reward-debug overlays."""

    if not (show_target_bodies or show_reward_debug):
        return MotionOverlaySelection(
            enabled=False,
            selected_indices=np.zeros((0,), dtype=np.int32),
        )

    if not (hasattr(env, "motion_loader") and hasattr(env, "motion_sampler")):
        log("WARNING: target/reward visualization only works for motion-tracking tasks.")
        return MotionOverlaySelection(
            enabled=False,
            selected_indices=np.zeros((0,), dtype=np.int32),
        )

    names = tuple(getattr(env.cfg, "body_names", ()))
    if len(names) == 0:
        log("WARNING: task has no body_names; cannot visualize targets.")
        return MotionOverlaySelection(
            enabled=False,
            selected_indices=np.zeros((0,), dtype=np.int32),
        )

    name_to_idx = {name: i for i, name in enumerate(names)}
    if target_body_names.strip():
        chosen = []
        for name in [n.strip() for n in target_body_names.split(",") if n.strip()]:
            if name in name_to_idx:
                chosen.append(name_to_idx[name])
            else:
                log(f"WARNING: body name not found in task body list: {name}")
        selected_indices = np.array(chosen, dtype=np.int32)
    else:
        selected_indices = np.arange(len(names), dtype=np.int32)

    if target_max_bodies > 0:
        selected_indices = selected_indices[:target_max_bodies]

    return MotionOverlaySelection(
        enabled=selected_indices.size > 0,
        selected_indices=selected_indices,
    )


__all__ = [
    "KeyboardCommander",
    "MotionOverlaySelection",
    "OffPolicyPlaybackSession",
    "PlaybackControls",
    "PlaybackSession",
    "RslRlPlaybackConfig",
    "RslRlPlaybackSession",
    "create_appo_playback_session",
    "create_distill_playback_session",
    "create_hora_distill_playback_session",
    "create_rsl_rl_playback_session",
    "create_sac_playback_session",
    "distill_command_intents_from_commands",
    "prepare_motion_overlay_selection",
    "select_torch_device",
]
