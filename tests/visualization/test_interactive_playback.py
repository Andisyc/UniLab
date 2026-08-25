from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import unilab.visualization.interactive_playback as interactive_playback

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from unilab.visualization.interactive_playback import (
    KeyboardCommander,
    OffPolicyPlaybackSession,
    PlaybackControls,
    RslRlPlaybackConfig,
    RslRlPlaybackSession,
    create_appo_playback_session,
    create_distill_playback_session,
    create_hora_distill_playback_session,
    create_rsl_rl_playback_session,
    create_sac_playback_session,
    distill_command_intents_from_commands,
    prepare_motion_overlay_selection,
)

_VEL_LIMIT = [[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]]


class _FakeWrappedEnv:
    def __init__(self, env: Any):
        self.env = env
        self.reset_calls = 0
        self.step_calls = 0
        self.last_actions = None

    def reset(self):
        self.reset_calls += 1
        return "obs", {}

    def step(self, actions):
        self.step_calls += 1
        self.last_actions = actions
        return f"obs_{self.step_calls}", 0.0, False, {}


def _fake_env(num_envs: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        action_space=SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        ),
        get_physics_state_snapshot=lambda: np.zeros((num_envs, 4), dtype=np.float32),
        state=SimpleNamespace(info={"motion_data": object()}),
    )


def test_playback_controls_gate_single_step_and_speed() -> None:
    controls = PlaybackControls(paused=True, speed=2.0)

    assert controls.consume_step_permission() is False
    controls.request_single_step()
    assert controls.consume_step_permission() is True
    assert controls.consume_step_permission() is False

    controls.resume()
    assert controls.consume_step_permission() is True

    controls.set_speed(0.0)
    assert controls.speed > 0.0
    controls.set_speed(4.0)
    assert controls.target_dt(0.02) == pytest.approx(0.005)


def test_playback_session_advance_respects_pause_and_single_step() -> None:
    env = _fake_env()
    wrapped = _FakeWrappedEnv(env)
    session = RslRlPlaybackSession(
        env=env,
        wrapped_env=wrapped,
        device="cpu",
        action_mode="zero",
        policy=None,
        num_envs=1,
    )
    controls = PlaybackControls(paused=True)

    session.reset()
    assert session.advance(controls) is False
    assert wrapped.step_calls == 0

    controls.request_single_step()
    assert session.advance(controls) is True
    assert wrapped.step_calls == 1
    assert torch.equal(wrapped.last_actions, torch.zeros((1, 2)))
    assert session.advance(controls) is False


def test_rt2_playback_session_set_external_command_refreshes_observation() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.refresh_calls = 0
            self.state = SimpleNamespace(
                info={"commands": np.zeros((1, 3), dtype=np.float32)},
                obs={"obs": np.zeros((1, 4), dtype=np.float32)},
            )

        def refresh_state(self) -> None:
            self.refresh_calls += 1
            self.state.obs["obs"][:, :3] = self.state.info["commands"][:, :3]

        def get_physics_state_snapshot(self):
            return np.zeros((1, 4), dtype=np.float32)

    class FakeWrapper:
        def __init__(self, env: FakeEnv) -> None:
            self.env = env

        def reset(self):
            return torch.zeros((1, 4), dtype=torch.float32), {}

        def get_observations(self):
            return torch.as_tensor(self.env.state.obs["obs"])

        def step(self, actions):
            return torch.zeros((1, 4), dtype=torch.float32), torch.zeros((1,)), False, {}

    env = FakeEnv()
    session = RslRlPlaybackSession(
        env=env,
        wrapped_env=FakeWrapper(env),
        device="cpu",
        action_mode="zero",
        policy=None,
        num_envs=1,
    )
    session.reset()
    session.set_external_command(np.asarray([0.2, 0.0, 0.0], dtype=np.float32))
    print(
        "[RT-2 command_sync_probe] "
        f"refresh_calls={env.refresh_calls} "
        f"session_obs_command={session.obs[0, :3].tolist()}"
    )

    assert env.refresh_calls == 1
    torch.testing.assert_close(session.obs[0, :3], torch.tensor([0.2, 0.0, 0.0]))


def test_offpolicy_playback_session_refreshes_observation_without_stepping() -> None:
    env = SimpleNamespace(
        state=SimpleNamespace(
            obs={"obs": np.asarray([[0.2, 0.0, 0.0, 1.0]], dtype=np.float32)},
            info={},
        ),
        get_physics_state_snapshot=lambda: np.zeros((1, 4), dtype=np.float32),
    )
    session = OffPolicyPlaybackSession(
        env=env,
        device="cpu",
        action_mode="zero",
        actor=None,
        actor_algo_type="sac",
        normalizer=None,
        num_envs=1,
        obs_extractor=lambda obs: obs["obs"],
        priv_info_resolver=lambda **_kwargs: None,
    )
    session.obs = np.zeros((1, 4), dtype=np.float32)

    refreshed = session.refresh_observation()

    np.testing.assert_array_equal(refreshed, env.state.obs["obs"])
    assert session.step_count == 0


def test_create_rsl_rl_playback_session_loads_checkpoint_and_runner_log_dir() -> None:
    env = SimpleNamespace(
        obs_groups_spec={"obs": 5},
        action_space=SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        ),
        get_physics_state_snapshot=lambda: np.zeros((1, 4), dtype=np.float32),
    )
    captured: dict[str, Any] = {}

    class Wrapper:
        def __init__(self, wrapped_env, *, device, policy_obs_mode):
            captured["wrapper_env"] = wrapped_env
            captured["device"] = device
            captured["policy_obs_mode"] = policy_obs_mode

        def reset(self):
            return "obs", {}

        def step(self, actions):
            return "obs", 0.0, False, {}

    class Runner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            captured["runner_log_dir"] = log_dir
            captured["train_cfg"] = train_cfg
            captured["runner_device"] = device

        def load(self, checkpoint, load_cfg):
            captured["checkpoint"] = checkpoint
            captured["load_cfg"] = load_cfg

        def get_inference_policy(self, *, device):
            captured["policy_device"] = device
            return lambda obs: torch.ones((1, 2))

    session, policy_obs_mode, checkpoint = create_rsl_rl_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="MyTask",
            load_run="-1",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="auto",
            algo_log_name="custom_ppo",
            log_root=None,
            num_envs=1,
        ),
        env_factory=lambda num_envs: env,
        algo_config={"runner": {"logger": "tensorboard"}},
        root_dir=Path("/repo"),
        device="cpu",
        checkpoint_resolver=lambda *args: "/tmp/model_10.pt",
        checkpoint_input_dim_reader=lambda path: 5,
        entrypoint_log_root=lambda root_dir, *, algo_log_name, log_root=None: (
            Path("/tmp") / algo_log_name
        ),
        wrapper_cls=Wrapper,
        runner_cls=Runner,
        policy_obs_dims_getter=lambda spec: (5, 7),
        train_cfg_normalizer=lambda cfg: cfg,
        log=lambda message: None,
    )

    assert session.env is env
    assert policy_obs_mode == "actor"
    assert checkpoint == "/tmp/model_10.pt"
    assert captured["runner_log_dir"] == "/tmp/custom_ppo/MyTask/play_temp"
    assert captured["checkpoint"] == "/tmp/model_10.pt"
    assert captured["train_cfg"]["runner"]["logger"] == "none"


def test_create_rsl_rl_playback_session_rejects_missing_env() -> None:
    with pytest.raises(RuntimeError, match="Playback env factory"):
        create_rsl_rl_playback_session(
            playback_cfg=RslRlPlaybackConfig(
                task="MyTask",
                load_run="-1",
                checkpoint=None,
                action_mode="zero",
                policy_obs_mode="auto",
                algo_log_name="custom_ppo",
                log_root=None,
                num_envs=1,
            ),
            env_factory=lambda num_envs: None,
            algo_config={},
            root_dir=Path("/repo"),
            device="cpu",
            checkpoint_resolver=lambda *args: None,
            checkpoint_input_dim_reader=lambda path: None,
            entrypoint_log_root=lambda root_dir, *, algo_log_name, log_root=None: Path("/tmp"),
            wrapper_cls=object,
            runner_cls=object,
            policy_obs_dims_getter=lambda spec: (0, 0),
            train_cfg_normalizer=lambda cfg: cfg,
            log=lambda message: None,
        )


def test_create_hora_distill_playback_session_loads_student_policy(tmp_path: Path) -> None:
    from tensordict import TensorDict

    checkpoint_path = tmp_path / "hora_stage2_last.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    runtime_cfg = SimpleNamespace(training=SimpleNamespace(task_name="RuntimeTask"))
    cfg = SimpleNamespace(
        training=SimpleNamespace(task_name="SharpaInhandRotation"),
        algo=SimpleNamespace(load_run=str(checkpoint_path)),
    )
    captured: dict[str, Any] = {}

    class FakeEnv:
        num_actions = 2
        action_space = SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        )
        state = SimpleNamespace(info={})

        def get_physics_state_snapshot(self):
            return np.zeros((1, 4), dtype=np.float32)

    class FakeWrapper:
        num_actions = 2

        def __init__(self, env: Any, *, device: str, policy_obs_mode: str):
            self.env = env
            captured["wrapper_env"] = env
            captured["wrapper_device"] = device
            captured["policy_obs_mode"] = policy_obs_mode

        def reset(self):
            obs = TensorDict(
                {
                    "actor": torch.ones((1, 3), dtype=torch.float32),
                    "proprio_hist": torch.ones((1, 2, 3), dtype=torch.float32),
                },
                batch_size=[1],
            )
            return obs, {}

        def step(self, actions):
            captured["actions"] = actions
            return self.reset()[0], torch.zeros((1,)), torch.zeros((1,), dtype=torch.bool), {}

    class FakeModule:
        def __init__(self, name: str):
            self.name = name
            self.eval_calls = 0

        def eval(self):
            self.eval_calls += 1

    actor = FakeModule("actor")
    hist_normalizer = FakeModule("hist_normalizer")

    def fake_student_policy(actor_obj, hist_obj, obs, *, device):
        captured["student_policy"] = (actor_obj, hist_obj, obs, device)
        return torch.full((1, 2), 0.5, dtype=torch.float32)

    deps = {
        "resolve_stage2_checkpoint_path": lambda cfg_obj: (checkpoint_path, checkpoint_path.parent),
        "get_log_root": lambda root_dir, cfg_obj: tmp_path / "logs",
        "format_stage2_play_checkpoint_error": lambda *args, **kwargs: "missing",
        "checkpoint_reader": lambda path, *, map_location, weights_only: {
            "model_state_dict": {},
            "distill_runtime_cfg": {"algo": {"model": {"hidden_dims": [8]}}},
        },
        "cfg_with_checkpoint_runtime": lambda cfg_obj, checkpoint: runtime_cfg,
        "build_play_env_cfg_override": lambda cfg_obj: {"env": "override"},
        "create_env": lambda cfg_obj, *, num_envs, env_cfg_override: FakeEnv(),
        "wrapper_cls": FakeWrapper,
        "build_student_actor_and_normalizer": lambda wrapped_env, cfg_obj, *, device: (
            actor,
            hist_normalizer,
        ),
        "load_distilled_checkpoint": lambda actor_obj, hist_obj, path, *, device: captured.update(
            {
                "loaded": (actor_obj, hist_obj, path, device),
            }
        ),
        "student_policy": fake_student_policy,
    }

    session, policy_obs_mode, checkpoint = create_hora_distill_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="SharpaInhandRotation",
            load_run=str(checkpoint_path),
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="hora_distill",
            log_root=None,
            num_envs=1,
        ),
        cfg=cfg,
        root_dir=tmp_path,
        device="cpu",
        deps=deps,
        log=lambda message: None,
    )

    assert session.env is captured["wrapper_env"]
    assert policy_obs_mode == "actor"
    assert checkpoint == str(checkpoint_path)
    assert captured["policy_obs_mode"] == "actor"
    assert captured["loaded"][0] is actor
    assert captured["loaded"][1] is hist_normalizer
    assert captured["loaded"][2] == checkpoint_path
    assert actor.eval_calls == 1
    assert hist_normalizer.eval_calls == 1

    session.reset()
    session.step_once()

    assert torch.equal(captured["actions"], torch.full((1, 2), 0.5))
    assert captured["student_policy"][0] is actor
    assert captured["student_policy"][1] is hist_normalizer
    assert str(captured["student_policy"][3]) == "cpu"


def test_create_hora_distill_playback_session_missing_checkpoint_uses_zero_actions(
    tmp_path: Path,
) -> None:
    runtime_cfg = SimpleNamespace(training=SimpleNamespace(task_name="RuntimeTask"))
    cfg = SimpleNamespace(
        training=SimpleNamespace(task_name="SharpaInhandRotation"),
        algo=SimpleNamespace(load_run="missing"),
    )
    captured: dict[str, Any] = {}
    messages: list[str] = []

    class FakeEnv:
        action_space = SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        )
        state = SimpleNamespace(info={})

        def get_physics_state_snapshot(self):
            return np.zeros((1, 4), dtype=np.float32)

    class FakeWrapper:
        def __init__(self, env: Any, *, device: str, policy_obs_mode: str):
            self.env = env
            captured["policy_obs_mode"] = policy_obs_mode

        def reset(self):
            return "obs", {}

        def step(self, actions):
            captured["actions"] = actions
            return "obs", torch.zeros((1,)), torch.zeros((1,), dtype=torch.bool), {}

    deps = {
        "resolve_stage2_checkpoint_path": lambda cfg_obj: (None, None),
        "get_log_root": lambda root_dir, cfg_obj: tmp_path / "logs",
        "format_stage2_play_checkpoint_error": lambda *args, **kwargs: "missing checkpoint",
        "apply_teacher_defaults": lambda cfg_obj: runtime_cfg,
        "build_play_env_cfg_override": lambda cfg_obj: {"env": "override"},
        "create_env": lambda cfg_obj, *, num_envs, env_cfg_override: FakeEnv(),
        "wrapper_cls": FakeWrapper,
    }

    session, policy_obs_mode, checkpoint = create_hora_distill_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="SharpaInhandRotation",
            load_run="missing",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="hora_distill",
            log_root=None,
            num_envs=1,
        ),
        cfg=cfg,
        root_dir=tmp_path,
        device="cpu",
        deps=deps,
        log=messages.append,
    )

    assert policy_obs_mode == "actor"
    assert checkpoint is None
    assert captured["policy_obs_mode"] == "actor"
    assert messages == [
        "missing checkpoint",
        "WARNING: falling back to zero actions.",
        "Policy obs mode: actor",
        "Action mode: policy",
    ]

    session.reset()
    session.step_once()

    assert torch.equal(captured["actions"], torch.zeros((1, 2)))


def test_create_distill_playback_session_loads_student_policy(tmp_path: Path) -> None:
    from unilab.algos.torch.distill import MLPStudentPolicy, save_distillation_checkpoint

    checkpoint_path = tmp_path / "generic_distill_student.pt"
    student = MLPStudentPolicy(obs_dim=5, action_dim=2, hidden_dims=(8,))
    save_distillation_checkpoint(
        checkpoint_path,
        student=student,
        agent_steps=4,
        teacher_metadata={"task_name": "G1WalkHeight"},
        distill_runtime_cfg={
            "student_obs_dim": 5,
            "student_action_dim": 2,
            "student_hidden_dims": [8],
            "student_activation": "elu",
            "student_squash_action": True,
        },
    )
    captured: dict[str, Any] = {}
    messages: list[str] = []

    class FakeEnv:
        action_space = SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        )
        state = SimpleNamespace(info={})

        def get_physics_state_snapshot(self):
            return np.zeros((1, 4), dtype=np.float32)

    class FakeWrapper:
        def __init__(self, env: Any, *, device: str, policy_obs_mode: str):
            self.env = env
            captured["wrapper_device"] = device
            captured["policy_obs_mode"] = policy_obs_mode

        def reset(self):
            return torch.ones((1, 5), dtype=torch.float32), {}

        def step(self, actions):
            captured["actions"] = actions
            return torch.ones((1, 5), dtype=torch.float32), torch.zeros((1,)), False, {}

    deps = {
        "resolve_checkpoint": lambda playback_cfg, cfg, root_dir: checkpoint_path,
        "create_env": lambda cfg, *, num_envs: FakeEnv(),
        "wrapper_cls": FakeWrapper,
    }

    session, policy_obs_mode, checkpoint = create_distill_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="G1WalkHeight",
            load_run=str(checkpoint_path),
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="distill",
            log_root=None,
            num_envs=1,
        ),
        cfg=SimpleNamespace(training=SimpleNamespace(task_name="G1WalkHeight")),
        root_dir=tmp_path,
        device="cpu",
        deps=deps,
        log=messages.append,
    )

    assert policy_obs_mode == "actor"
    assert checkpoint == str(checkpoint_path)
    assert captured["policy_obs_mode"] == "actor"
    assert messages == [
        f"Loading distillation student checkpoint: {checkpoint_path}",
        (
            "Distill checkpoint diagnostics: student_obs_dim=5, "
            "student_action_dim=2, agent_steps=4, obs_normalizer=absent"
        ),
        "Policy obs mode: actor",
        "Action mode: policy",
    ]

    session.reset()
    session.step_once()

    assert captured["actions"].shape == (1, 2)
    assert captured["actions"].requires_grad is False
    assert torch.isfinite(captured["actions"]).all()


def test_distill_command_intents_from_commands_thresholds() -> None:
    intents = distill_command_intents_from_commands(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.06, 0.0, 0.0],
                [0.0, 0.0, -0.06],
                [0.05, 0.0, 0.05],
            ],
            dtype=np.float32,
        ),
        xy_threshold=0.05,
        yaw_threshold=0.05,
    )

    assert intents == ("inactive", "active", "active", "inactive")


def test_distill_playback_hard_routes_moe_by_command_intent(tmp_path: Path) -> None:
    from unilab.algos.torch.distill import MoEStudentPolicy, save_distillation_checkpoint

    checkpoint_path = tmp_path / "moe_student.pt"
    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=2,
        num_experts=2,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        squash_action=False,
    )
    with torch.no_grad():
        for param in student.parameters():
            param.zero_()
        student.experts[0].net[-1].bias.fill_(0.25)
        student.experts[1].net[-1].bias.fill_(-0.4)
        student.router[-1].bias.copy_(torch.tensor([5.0, -5.0]))
    save_distillation_checkpoint(
        checkpoint_path,
        student=student,
        agent_steps=8,
        teacher_metadata={"task_name": "G1WalkFlat"},
        distill_runtime_cfg={
            "student_model_type": "moe",
            "student_obs_dim": 4,
            "student_action_dim": 2,
            "student_num_experts": 2,
            "student_expert_hidden_dims": [],
            "student_router_hidden_dims": [],
            "student_activation": "elu",
            "student_squash_action": False,
            "student_routing_mode": "soft",
            "student_router_temperature": 1.0,
            "command_intent_loss_coef": 0.0,
            "command_intent_expert_targets": {"active": 0, "inactive": 1},
            "expert_behavior_loss_source": "command_intent",
        },
    )
    captured: dict[str, Any] = {}

    class FakeEnv:
        action_space = SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        )

        def __init__(self) -> None:
            self.state = SimpleNamespace(info={"commands": np.zeros((1, 3), dtype=np.float32)})

        def get_physics_state_snapshot(self):
            return np.zeros((1, 4), dtype=np.float32)

    class FakeWrapper:
        def __init__(self, env: Any, *, device: str, policy_obs_mode: str):
            self.env = env

        def reset(self):
            return torch.zeros((1, 4), dtype=torch.float32), {}

        def step(self, actions):
            captured["actions"] = actions.detach().clone()
            return torch.zeros((1, 4), dtype=torch.float32), torch.zeros((1,)), False, {}

    deps = {
        "resolve_checkpoint": lambda playback_cfg, cfg, root_dir: checkpoint_path,
        "create_env": lambda cfg, *, num_envs: FakeEnv(),
        "wrapper_cls": FakeWrapper,
    }
    cfg = SimpleNamespace(
        training=SimpleNamespace(task_name="G1WalkFlat"),
        interactive=SimpleNamespace(
            distill_command_routing="auto",
            distill_command_xy_threshold=0.05,
            distill_command_yaw_threshold=0.05,
            distill_command_routing_bias=10.0,
        ),
        algo=SimpleNamespace(
            command_intent_expert_targets={"active": 0, "inactive": 1},
        ),
    )

    session, _policy_obs_mode, _checkpoint = create_distill_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="G1WalkFlat",
            load_run=str(checkpoint_path),
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="distill",
            log_root=None,
            num_envs=1,
        ),
        cfg=cfg,
        root_dir=tmp_path,
        device="cpu",
        deps=deps,
        log=lambda message: None,
    )

    session.reset()
    session.step_once()
    assert torch.allclose(captured["actions"], torch.full((1, 2), -0.4))
    assert getattr(session.policy, "_unilab_distill_last_command_intents") == ("inactive",)
    assert getattr(session.policy, "_unilab_distill_last_expected_experts") == (1,)
    assert getattr(session.policy, "_unilab_distill_last_selected_experts") == (1,)
    assert getattr(session.policy, "_unilab_distill_command_routing_applied") is True

    session.env.state.info["commands"] = np.array([[0.2, 0.0, 0.0]], dtype=np.float32)
    session.step_once()
    assert torch.allclose(captured["actions"], torch.full((1, 2), 0.25))
    assert getattr(session.policy, "_unilab_distill_last_command_intents") == ("active",)
    assert getattr(session.policy, "_unilab_distill_last_expected_experts") == (0,)


def test_rt1_playback_probe_exposes_command_observation_skew(tmp_path: Path) -> None:
    """Expose route=active while the session still feeds inactive obs."""

    from unilab.algos.torch.distill import MoEStudentPolicy, save_distillation_checkpoint

    checkpoint_path = tmp_path / "moe_student.pt"
    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=2,
        num_experts=2,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        squash_action=False,
    )
    with torch.no_grad():
        for param in student.parameters():
            param.zero_()
        student.router[-1].bias.copy_(torch.tensor([5.0, -5.0]))
        student.experts[0].net[-1].weight[0, 0] = 1.0
    save_distillation_checkpoint(
        checkpoint_path,
        student=student,
        agent_steps=8,
        teacher_metadata={"task_name": "G1WalkFlat"},
        distill_runtime_cfg={
            "student_model_type": "moe",
            "student_obs_dim": 4,
            "student_action_dim": 2,
            "student_num_experts": 2,
            "student_expert_hidden_dims": [],
            "student_router_hidden_dims": [],
            "student_activation": "elu",
            "student_squash_action": False,
            "student_routing_mode": "soft",
            "student_router_temperature": 1.0,
            "command_intent_loss_coef": 0.0,
            "command_intent_expert_targets": {"active": 0, "inactive": 1},
            "expert_behavior_loss_source": "command_intent",
        },
    )
    captured: dict[str, Any] = {}

    class FakeEnv:
        action_space = SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        )

        def __init__(self) -> None:
            self.state = SimpleNamespace(info={"commands": np.zeros((1, 3), dtype=np.float32)})

        def get_physics_state_snapshot(self):
            return np.zeros((1, 4), dtype=np.float32)

    class FakeWrapper:
        def __init__(self, env: Any, *, device: str, policy_obs_mode: str):
            self.env = env

        def reset(self):
            return torch.zeros((1, 4), dtype=torch.float32), {}

        def step(self, actions):
            captured["actions"] = actions.detach().clone()
            return torch.zeros((1, 4), dtype=torch.float32), torch.zeros((1,)), False, {}

    deps = {
        "resolve_checkpoint": lambda playback_cfg, cfg, root_dir: checkpoint_path,
        "create_env": lambda cfg, *, num_envs: FakeEnv(),
        "wrapper_cls": FakeWrapper,
    }
    cfg = SimpleNamespace(
        training=SimpleNamespace(task_name="G1WalkFlat"),
        interactive=SimpleNamespace(
            distill_command_routing="hard",
            distill_command_xy_threshold=0.05,
            distill_command_yaw_threshold=0.05,
            distill_command_routing_bias=10.0,
        ),
        algo=SimpleNamespace(
            command_intent_expert_targets={"active": 0, "inactive": 1},
        ),
    )
    session, _policy_obs_mode, _checkpoint = create_distill_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="G1WalkFlat",
            load_run=str(checkpoint_path),
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="distill",
            log_root=None,
            num_envs=1,
        ),
        cfg=cfg,
        root_dir=tmp_path,
        device="cpu",
        deps=deps,
        log=lambda message: None,
    )

    session.reset()
    cached_obs_command = session.obs[0, :3].detach().clone()
    session.env.state.info["commands"] = np.asarray([[0.2, 0.0, 0.0]], dtype=np.float32)
    runtime_command = torch.tensor(session.env.state.info["commands"], dtype=torch.float32)
    session.step_once()
    print(
        "[RT-1 command_obs_probe] "
        f"routing={getattr(session.policy, '_unilab_distill_last_command_intents')} "
        f"runtime_command={runtime_command[0].tolist()} "
        f"cached_obs_command={cached_obs_command.tolist()} "
        f"action={captured['actions'][0].tolist()}"
    )

    assert getattr(session.policy, "_unilab_distill_last_command_intents") == ("active",)
    assert not torch.allclose(cached_obs_command, runtime_command[0])
    assert torch.allclose(captured["actions"][0], torch.zeros(2))


def test_distill_playback_resolves_explicit_student_checkpoint_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import unilab.training.run as training_run

    checkpoint_path = tmp_path / "student.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    cfg = SimpleNamespace(training=SimpleNamespace(task_name="G1WalkFlat"))

    def fail_resolve_task_checkpoint_path(*args, **kwargs):
        raise AssertionError("training.play_checkpoint_path must bypass run/checkpoint resolution")

    monkeypatch.setattr(
        training_run,
        "resolve_task_checkpoint_path",
        fail_resolve_task_checkpoint_path,
    )

    resolved = interactive_playback._resolve_distill_checkpoint_from_playback_cfg(
        RslRlPlaybackConfig(
            task="G1WalkFlat",
            load_run="-1",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="distill",
            log_root=None,
            checkpoint_path=str(checkpoint_path),
        ),
        cfg,
        tmp_path,
    )

    assert resolved == checkpoint_path


def test_create_distill_playback_session_missing_checkpoint_uses_zero_actions(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    messages: list[str] = []

    class FakeEnv:
        action_space = SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        )
        state = SimpleNamespace(info={})

        def get_physics_state_snapshot(self):
            return np.zeros((1, 4), dtype=np.float32)

    class FakeWrapper:
        def __init__(self, env: Any, *, device: str, policy_obs_mode: str):
            self.env = env
            captured["policy_obs_mode"] = policy_obs_mode

        def reset(self):
            return torch.ones((1, 5), dtype=torch.float32), {}

        def step(self, actions):
            captured["actions"] = actions
            return torch.ones((1, 5), dtype=torch.float32), torch.zeros((1,)), False, {}

    deps = {
        "resolve_checkpoint": lambda playback_cfg, cfg, root_dir: None,
        "create_env": lambda cfg, *, num_envs: FakeEnv(),
        "wrapper_cls": FakeWrapper,
    }

    session, policy_obs_mode, checkpoint = create_distill_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="G1WalkHeight",
            load_run="missing",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="distill",
            log_root=None,
            num_envs=1,
        ),
        cfg=SimpleNamespace(training=SimpleNamespace(task_name="G1WalkHeight")),
        root_dir=tmp_path,
        device="cpu",
        deps=deps,
        log=messages.append,
    )

    assert policy_obs_mode == "actor"
    assert checkpoint is None
    assert captured["policy_obs_mode"] == "actor"
    assert messages == [
        "WARNING: no generic distillation student checkpoint found - falling back to zero actions.",
        "Policy obs mode: actor",
        "Action mode: policy",
    ]

    session.reset()
    session.step_once()

    assert torch.equal(captured["actions"], torch.zeros((1, 2)))


def test_keyboard_commander_nudges_stack_and_clamp_to_vel_limit() -> None:
    commander = KeyboardCommander.from_vel_limit(_VEL_LIMIT, step_lin=0.1, step_ang=0.2)
    assert commander.command.tolist() == [0.0, 0.0, 0.0]

    # Linear and angular axes stack independently.
    commander.nudge(KeyboardCommander.AXIS_VX, +1.0)
    commander.nudge(KeyboardCommander.AXIS_VYAW, +1.0)
    assert commander.command == pytest.approx([0.1, 0.0, 0.2])

    # Repeated nudges saturate at the configured velocity limits.
    for _ in range(50):
        commander.nudge(KeyboardCommander.AXIS_VX, +1.0)
        commander.nudge(KeyboardCommander.AXIS_VY, -1.0)
    assert commander.command[0] == pytest.approx(1.0)
    assert commander.command[1] == pytest.approx(-0.4)

    commander.zero()
    assert commander.command.tolist() == [0.0, 0.0, 0.0]


def test_keyboard_commander_rejects_bad_vel_limit_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(2, 3\)"):
        KeyboardCommander.from_vel_limit([[0.0, 0.0], [1.0, 1.0]])


def test_prepare_motion_overlay_selection_filters_body_names() -> None:
    env = SimpleNamespace(
        motion_loader=object(),
        motion_sampler=object(),
        cfg=SimpleNamespace(body_names=("base", "left_foot", "right_foot")),
    )
    messages: list[str] = []

    selection = prepare_motion_overlay_selection(
        env,
        show_target_bodies=True,
        show_reward_debug=False,
        target_body_names="right_foot,missing,base",
        target_max_bodies=1,
        log=messages.append,
    )

    assert selection.enabled is True
    assert selection.selected_indices.tolist() == [2]
    assert messages == ["WARNING: body name not found in task body list: missing"]


def test_appo_hora_playback_session_uses_hora_wrapper_and_actor_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import rsl_rl.utils as rsl_rl_utils
    from omegaconf import OmegaConf
    from tensordict import TensorDict

    import unilab.algos.torch.hora.models as hora_models
    import unilab.algos.torch.hora.rsl_rl as hora_rsl

    checkpoint = tmp_path / "model_10.pt"
    torch.save({"actor": {"weight": torch.tensor(1.0)}}, checkpoint)
    captured: dict[str, Any] = {}

    class FakeHoraWrapper:
        def __init__(self, env, *, device, policy_obs_mode):
            captured["wrapper_cls"] = "hora"
            captured["policy_obs_mode"] = policy_obs_mode
            self.env = env
            self.device = device
            self.num_envs = env.num_envs

        def get_observations(self):
            return TensorDict(
                {
                    "actor": torch.zeros((1, 3)),
                    "priv_info": torch.zeros((1, 2)),
                    "proprio_hist": torch.zeros((1, 4, 3)),
                },
                batch_size=1,
            )

        def reset(self):
            return self.get_observations(), {}

        def step(self, actions):
            captured["step_actions"] = actions
            return self.get_observations(), torch.zeros(1), torch.zeros(1).bool(), {}

    class FakeActor(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["actor_kwargs"] = kwargs

        def load_state_dict(self, state_dict, strict=True):
            captured["loaded_actor"] = state_dict
            return None

        def forward(self, obs):
            captured["policy_obs"] = obs
            return torch.ones((1, 2), dtype=torch.float32)

    fake_env = SimpleNamespace(
        num_envs=1,
        obs_groups_spec={"obs": 3, "critic": 5},
        action_space=SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        ),
        get_physics_state_snapshot=lambda: np.zeros((1, 4), dtype=np.float32),
        state=SimpleNamespace(info={}),
    )
    cfg = OmegaConf.create(
        {
            "training": {"task_name": "Task", "play_env_num": 1, "log_root": None},
            "algo": {
                "algo_log_name": "hora_appo",
                "load_run": str(tmp_path),
                "checkpoint": str(checkpoint),
                "runtime_impl": "hora_appo",
                "obs_groups": {"actor": {"actor": 0, "priv_info": 0}},
                "actor": {"class_name": "fake.Actor"},
                "critic": {},
            },
        }
    )

    monkeypatch.setattr(hora_rsl, "HoraRslRlVecEnvWrapper", FakeHoraWrapper)
    monkeypatch.setattr(rsl_rl_utils, "resolve_callable", lambda path: FakeActor)
    monkeypatch.setattr(
        hora_models,
        "build_hora_shared_actor_critic",
        lambda **kwargs: torch.nn.Identity(),
    )

    session, policy_obs_mode, resolved_checkpoint = create_appo_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="Task",
            load_run="run",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="auto",
            algo_log_name="hora_appo",
            log_root=None,
        ),
        cfg=cfg,
        rl_cfg=OmegaConf.to_container(cfg.algo, resolve=True),
        env_factory=lambda num_envs: fake_env,
        root_dir=tmp_path,
        device="cpu",
        wrapper_cls=object,
        log=lambda message: None,
    )

    session.reset()
    assert session.advance(PlaybackControls()) is True
    assert policy_obs_mode == "actor"
    assert resolved_checkpoint == str(checkpoint)
    assert captured["wrapper_cls"] == "hora"
    assert captured["policy_obs_mode"] == "actor"
    torch.testing.assert_close(captured["loaded_actor"]["weight"], torch.tensor(1.0))
    assert isinstance(captured["policy_obs"], TensorDict)
    torch.testing.assert_close(captured["step_actions"], torch.ones((1, 2)))


def test_sac_hora_playback_session_updates_priv_info_after_reset_and_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import train_offpolicy
    from omegaconf import OmegaConf

    import unilab.algos.torch.common.actor_factory as actor_factory

    checkpoint = tmp_path / "model_10.pt"
    torch.save({"actor": {}}, checkpoint)
    reset_priv = np.array([[4.0, 5.0]], dtype=np.float32)
    step_priv = np.array([[8.0, 9.0]], dtype=np.float32)
    captured: dict[str, Any] = {}

    class FakeActor:
        def eval(self):
            return self

        def load_state_dict(self, state_dict):
            captured["loaded_actor"] = state_dict

        def explore(self, obs, priv_info, deterministic=True):
            captured["reset_priv_info"] = priv_info.detach().cpu().numpy()
            captured["deterministic"] = deterministic
            return torch.zeros((obs.shape[0], 2), dtype=obs.dtype, device=obs.device)

    class FakeEnv:
        num_envs = 1
        obs_groups_spec = {"obs": 3, "critic": 5}
        action_space = SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        )
        state = None

        def init_state(self):
            self.state = SimpleNamespace(info={})

        def reset(self, env_indices):
            del env_indices
            return (
                {
                    "obs": np.zeros((1, 3), dtype=np.float32),
                    "critic": np.concatenate(
                        [np.zeros((1, 3), dtype=np.float32), reset_priv],
                        axis=1,
                    ),
                },
                {"critic_info": reset_priv},
            )

        def step(self, actions):
            captured["actions"] = actions
            self.state = SimpleNamespace(
                obs={
                    "obs": np.ones((1, 3), dtype=np.float32),
                    "critic": np.concatenate(
                        [np.ones((1, 3), dtype=np.float32), step_priv],
                        axis=1,
                    ),
                },
                info={"critic_info": step_priv},
            )
            return self.state

        def get_physics_state_snapshot(self):
            return np.zeros((1, 4), dtype=np.float32)

    cfg = OmegaConf.create(
        {
            "training": {"task_name": "Task", "device": None},
            "algo": {
                "algo_log_name": "hora_sac",
                "load_run": "run",
                "actor_hidden_dim": 16,
                "use_layer_norm": False,
                "runtime_impl": "hora_sac",
            },
        }
    )

    monkeypatch.setattr(
        train_offpolicy,
        "default_device",
        lambda torch_module, preferred=None: "cpu",
    )
    monkeypatch.setattr(train_offpolicy, "resolve_play_obs_dims", lambda spec: (3, 5))
    monkeypatch.setattr(
        train_offpolicy,
        "resolve_play_actor_spec",
        lambda algo_name, cfg, *, obs_dim, critic_obs_dim: (
            "hora_sac",
            {"priv_info_dim": 2},
        ),
    )
    monkeypatch.setattr(
        train_offpolicy,
        "resolve_checkpoint_path",
        lambda *args, **kwargs: (str(checkpoint), str(tmp_path)),
    )
    monkeypatch.setattr(actor_factory, "build_actor", lambda *args, **kwargs: FakeActor())

    session, policy_obs_mode, resolved_checkpoint = create_sac_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="Task",
            load_run="run",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="hora_sac",
            log_root=None,
        ),
        cfg=cfg,
        env_factory=lambda num_envs: FakeEnv(),
        root_dir=tmp_path,
        device="cpu",
        log=lambda message: None,
    )

    session.reset()
    assert session.advance(PlaybackControls()) is True
    assert policy_obs_mode == "actor"
    assert resolved_checkpoint == str(checkpoint)
    assert captured["loaded_actor"] == {}
    np.testing.assert_allclose(captured["reset_priv_info"], reset_priv)
    np.testing.assert_allclose(session.current_priv_info, step_priv)
    assert captured["deterministic"] is True


def test_sac_playback_rejects_checkpoint_obs_dim_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import train_offpolicy
    from omegaconf import OmegaConf

    import unilab.algos.torch.common.actor_factory as actor_factory

    checkpoint = tmp_path / "model_10.pt"
    torch.save({"actor": {"net.0.weight": torch.zeros((512, 99))}}, checkpoint)

    class FakeEnv:
        num_envs = 1
        obs_groups_spec = {"obs": 98}
        action_space = SimpleNamespace(shape=(29,))
        state = SimpleNamespace(info={})

    class FakeActor(torch.nn.Module):
        def __init__(self):
            super().__init__()

    cfg = OmegaConf.create(
        {
            "training": {"task_name": "G1WalkFlat", "device": None},
            "algo": {
                "algo_log_name": "fast_sac",
                "load_run": "run",
                "actor_hidden_dim": 16,
                "use_layer_norm": False,
                "runtime_impl": "sac",
                "obs_normalization": False,
            },
        }
    )

    monkeypatch.setattr(
        train_offpolicy, "default_device", lambda torch_module, preferred=None: "cpu"
    )
    monkeypatch.setattr(train_offpolicy, "resolve_play_obs_dims", lambda spec: (98, 98))
    monkeypatch.setattr(
        train_offpolicy,
        "resolve_play_actor_spec",
        lambda algo_name, cfg, *, obs_dim, critic_obs_dim: ("sac", {}),
    )
    monkeypatch.setattr(
        train_offpolicy,
        "resolve_checkpoint_path",
        lambda *args, **kwargs: (str(checkpoint), str(tmp_path)),
    )
    monkeypatch.setattr(actor_factory, "build_actor", lambda *args, **kwargs: FakeActor())

    with pytest.raises(RuntimeError, match="checkpoint=99, playback_env_obs=98"):
        create_sac_playback_session(
            playback_cfg=RslRlPlaybackConfig(
                task="G1WalkFlat",
                load_run="run",
                checkpoint=None,
                action_mode="policy",
                policy_obs_mode="actor",
                algo_log_name="fast_sac",
                log_root=None,
            ),
            cfg=cfg,
            env_factory=lambda num_envs: FakeEnv(),
            root_dir=tmp_path,
            device="cpu",
            log=lambda message: None,
        )


def test_sac_playback_loads_legacy_tar_checkpoint_with_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import train_offpolicy
    from omegaconf import OmegaConf

    import unilab.algos.torch.common.actor_factory as actor_factory
    import unilab.visualization.interactive_playback as playback

    checkpoint = tmp_path / "model_10.pt"
    checkpoint.write_bytes(b"legacy-tar-placeholder")
    captured: dict[str, Any] = {"logs": [], "load_calls": []}

    class FakeEnv:
        num_envs = 1
        obs_groups_spec = {"obs": 3}
        action_space = SimpleNamespace(shape=(2,))
        state = SimpleNamespace(info={})

    class FakeActor:
        def eval(self):
            return self

        def load_state_dict(self, state_dict):
            captured["loaded_actor"] = state_dict

    cfg = OmegaConf.create(
        {
            "training": {"task_name": "Task", "device": None},
            "algo": {
                "algo_log_name": "fast_sac",
                "load_run": "run",
                "actor_hidden_dim": 16,
                "use_layer_norm": False,
                "runtime_impl": "sac",
                "obs_normalization": False,
            },
        }
    )

    def fake_torch_load(path, *, map_location=None, weights_only=True):
        captured["load_calls"].append(
            {
                "path": str(path),
                "map_location": map_location,
                "weights_only": weights_only,
            }
        )
        if weights_only:
            raise RuntimeError(
                "Cannot use ``weights_only=True`` with files saved in the legacy .tar format."
            )
        return {"actor": {"net.0.weight": torch.zeros((4, 3))}}

    monkeypatch.setattr(
        train_offpolicy, "default_device", lambda torch_module, preferred=None: "cpu"
    )
    monkeypatch.setattr(train_offpolicy, "resolve_play_obs_dims", lambda spec: (3, 3))
    monkeypatch.setattr(
        train_offpolicy,
        "resolve_play_actor_spec",
        lambda algo_name, cfg, *, obs_dim, critic_obs_dim: ("sac", {}),
    )
    monkeypatch.setattr(
        train_offpolicy,
        "resolve_checkpoint_path",
        lambda *args, **kwargs: (str(checkpoint), str(tmp_path)),
    )
    monkeypatch.setattr(actor_factory, "build_actor", lambda *args, **kwargs: FakeActor())
    monkeypatch.setattr(playback.torch, "load", fake_torch_load)

    session, policy_obs_mode, resolved_checkpoint = create_sac_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="Task",
            load_run="run",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="fast_sac",
            log_root=None,
        ),
        cfg=cfg,
        env_factory=lambda num_envs: FakeEnv(),
        root_dir=tmp_path,
        device="cpu",
        log=lambda message: captured["logs"].append(message),
    )

    assert session.actor is not None
    assert policy_obs_mode == "actor"
    assert resolved_checkpoint == str(checkpoint)
    assert [call["weights_only"] for call in captured["load_calls"]] == [True, False]
    assert captured["loaded_actor"]["net.0.weight"].shape == (4, 3)
    assert any("legacy PyTorch .tar serialization" in message for message in captured["logs"])


def test_sac_playback_reports_corrupt_legacy_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import unilab.visualization.interactive_playback as playback

    checkpoint = tmp_path / "model_10.pt"
    checkpoint.write_bytes(b"corrupt")
    captured: dict[str, Any] = {"load_calls": []}

    def fake_torch_load(path, *, map_location=None, weights_only=True):
        captured["load_calls"].append(weights_only)
        if weights_only:
            raise RuntimeError(
                "Cannot use ``weights_only=True`` with files saved in the legacy .tar format."
            )
        raise KeyError("storages")

    monkeypatch.setattr(playback.torch, "load", fake_torch_load)

    with pytest.raises(RuntimeError, match="corrupted, incomplete, or not a PyTorch checkpoint"):
        playback._load_playback_checkpoint(
            str(checkpoint), device_name="cpu", log=lambda message: None
        )
    assert captured["load_calls"] == [True, False]


def test_hora_distill_playback_session_loads_stage2_checkpoint_and_student_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import train_hora_distill
    from omegaconf import OmegaConf
    from tensordict import TensorDict

    import unilab.algos.torch.hora.distill as distill
    import unilab.algos.torch.hora.rsl_rl as hora_rsl
    import unilab.training as training

    checkpoint = tmp_path / "hora_stage2_last.pt"
    torch.save({"model_state_dict": {}, "distill_runtime_cfg": {}}, checkpoint)
    captured: dict[str, Any] = {}

    class FakeWrapper:
        def __init__(self, env, *, device, policy_obs_mode):
            self.env = env
            self.device = device
            self.num_envs = env.num_envs
            self.num_actions = 2
            captured["policy_obs_mode"] = policy_obs_mode

        def get_observations(self):
            return TensorDict(
                {
                    "actor": torch.zeros((1, 3)),
                    "priv_info": torch.zeros((1, 2)),
                    "proprio_hist": torch.zeros((1, 4, 3)),
                },
                batch_size=1,
            )

        def reset(self):
            return self.get_observations(), {}

        def step(self, actions):
            captured["actions"] = actions
            return self.get_observations(), torch.zeros(1), torch.zeros(1).bool(), {}

    fake_env = SimpleNamespace(
        num_envs=1,
        action_space=SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        ),
        get_physics_state_snapshot=lambda: np.zeros((1, 4), dtype=np.float32),
        state=SimpleNamespace(info={}),
    )
    cfg = OmegaConf.create(
        {
            "training": {"task_name": "Task", "sim_backend": "mujoco", "log_root": None},
            "algo": {"algo_log_name": "hora_distill", "load_run": "run", "checkpoint": -1},
        }
    )

    monkeypatch.setattr(
        train_hora_distill,
        "_resolve_stage2_checkpoint_path",
        lambda cfg: (checkpoint, tmp_path),
    )

    def fake_cfg_with_checkpoint_runtime(cfg, checkpoint_payload):
        captured["runtime_helper_checkpoint"] = checkpoint_payload
        return cfg

    monkeypatch.setattr(
        train_hora_distill,
        "_cfg_with_checkpoint_runtime",
        fake_cfg_with_checkpoint_runtime,
    )
    monkeypatch.setattr(train_hora_distill, "_build_play_env_cfg_override", lambda cfg: {})
    monkeypatch.setattr(
        train_hora_distill,
        "_student_policy",
        lambda actor, hist_normalizer, obs, *, device: torch.ones((1, 2)),
    )
    monkeypatch.setattr(training, "create_env", lambda *args, **kwargs: fake_env)
    monkeypatch.setattr(hora_rsl, "HoraRslRlVecEnvWrapper", FakeWrapper)
    monkeypatch.setattr(
        distill,
        "build_student_actor_and_normalizer",
        lambda wrapped_env, cfg, *, device: (
            SimpleNamespace(eval=lambda: None),
            SimpleNamespace(eval=lambda: None),
        ),
    )

    def fake_load_distilled_checkpoint(actor, hist_normalizer, checkpoint_path, *, device):
        captured["loaded_checkpoint"] = checkpoint_path
        return {}

    monkeypatch.setattr(distill, "load_distilled_checkpoint", fake_load_distilled_checkpoint)

    session, policy_obs_mode, resolved_checkpoint = create_hora_distill_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="Task",
            load_run="run",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="hora_distill",
            log_root=None,
        ),
        cfg=cfg,
        root_dir=tmp_path,
        device="cpu",
        log=lambda message: None,
    )

    session.reset()
    assert session.advance(PlaybackControls()) is True
    assert policy_obs_mode == "actor"
    assert resolved_checkpoint == str(checkpoint)
    assert captured["runtime_helper_checkpoint"] == {
        "model_state_dict": {},
        "distill_runtime_cfg": {},
    }
    assert captured["policy_obs_mode"] == "actor"
    assert captured["loaded_checkpoint"] == checkpoint
    torch.testing.assert_close(captured["actions"], torch.ones((1, 2)))
