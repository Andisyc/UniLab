from __future__ import annotations

from pathlib import Path

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from unilab.algos.torch.amp import learner as amp_learner
from unilab.algos.torch.amp.learner import AMPAPPOLearner, AMPDiscriminator
from unilab.algos.torch.appo.checkpoint import load_appo_checkpoint, save_appo_checkpoint

_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "unilab"
    / "assets"
    / "motions"
    / "g1"
    / "amp_walk"
    / "manifest.json"
)


def _models() -> tuple[MLPModel, MLPModel]:
    obs_groups = {"actor": {"policy": 4}, "critic": {"policy": 5}}
    actor_example = TensorDict({"policy": torch.zeros(4, 4)}, batch_size=4)
    critic_example = TensorDict({"policy": torch.zeros(4, 5)}, batch_size=4)
    actor = MLPModel(
        actor_example,
        obs_groups,
        "actor",
        2,
        hidden_dims=[16],
        activation="elu",
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": 0.5,
            "std_type": "scalar",
        },
    )
    critic = MLPModel(
        critic_example,
        obs_groups,
        "critic",
        1,
        hidden_dims=[16],
        activation="elu",
    )
    return actor, critic


def _learner(seed: int = 7) -> AMPAPPOLearner:
    actor, critic = _models()
    return AMPAPPOLearner(
        actor=actor,
        critic=critic,
        device="cpu",
        num_learning_epochs=1,
        num_mini_batches=2,
        learning_rate=1e-3,
        enable_compile=False,
        amp_motion_manifest=_MANIFEST,
        amp_hidden_dims=[16, 8],
        amp_reward_coef=0.1,
        amp_task_reward_lerp=0.75,
        amp_replay_capacity=32,
        amp_discriminator_batch_size=8,
        amp_discriminator_updates=1,
        amp_discriminator_learning_rate=1e-3,
        amp_gradient_penalty=1.0,
        amp_seed=seed,
    )


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(99)
    t, n = 2, 4
    return {
        "observations": torch.randn(t, n, 4, generator=generator),
        "critic": torch.randn(t, n, 5, generator=generator),
        "actions": torch.randn(t, n, 2, generator=generator),
        "actions_log_prob": torch.zeros(t, n),
        "rewards": torch.linspace(0.2, 1.0, t * n).reshape(t, n),
        "dones": torch.zeros(t, n),
        "truncated": torch.zeros(t, n),
        "last_obs": torch.randn(n, 4, generator=generator),
        "last_critic": torch.randn(n, 5, generator=generator),
        "amp_state": torch.randn(t, n, 195, generator=generator),
        "amp_next_state": torch.randn(t, n, 195, generator=generator),
    }


def test_discriminator_reward_matches_amp_mjlab_formula() -> None:
    discriminator = AMPDiscriminator(390, [4], reward_coef=0.1)
    with torch.no_grad():
        for parameter in discriminator.parameters():
            parameter.zero_()
        discriminator.head.bias.fill_(0.5)

    state = torch.zeros(3, 195)
    reward, logits = discriminator.predict_reward(state, state)

    expected = 0.1 * max(1.0 - 0.25 * (0.5 - 1.0) ** 2, 0.0)
    torch.testing.assert_close(logits, torch.full((3,), 0.5))
    torch.testing.assert_close(reward, torch.full((3,), expected))


def test_policy_health_metrics_expose_zero_plateau_and_reward_authority() -> None:
    logits = torch.tensor([-1.2, -1.0, -0.95, 0.0, 1.0])
    style_reward = 0.1 * torch.clamp(1.0 - 0.25 * torch.square(logits - 1.0), min=0.0)
    task_reward = torch.full((5,), 0.2)

    metrics = amp_learner.amp_policy_health_metrics(
        logits=logits,
        style_reward=style_reward,
        task_reward=task_reward,
        task_reward_lerp=0.75,
        expert_motion_count=2,
        expert_transition_count=935,
        expert_draw_count=8,
    )

    assert metrics["amp/policy_logit_p50"] == pytest.approx(-0.95)
    assert metrics["amp/policy_zero_style_fraction"] == pytest.approx(0.4)
    assert metrics["amp/task_weighted_mean"] == pytest.approx(0.15)
    assert metrics["amp/style_weighted_mean"] == pytest.approx(
        0.25 * float(style_reward.mean())
    )
    assert metrics["amp/expert_motion_count"] == 2.0
    assert metrics["amp/expert_transition_count"] == 935.0
    assert metrics["amp/expert_draw_count"] == 8.0


def test_frozen_discriminator_reward_enters_vtrace_before_updates() -> None:
    learner = _learner()
    batch = _batch()
    task_reward = batch["rewards"].clone()
    with torch.no_grad():
        amp_reward, _ = learner.predict_amp_reward(batch["amp_state"], batch["amp_next_state"])
    expected = 0.75 * task_reward + 0.25 * amp_reward

    learner.process_batch(batch)

    torch.testing.assert_close(batch["rewards"], expected)
    torch.testing.assert_close(batch["_amp_task_rewards"], task_reward)
    assert learner.discriminator_version == 0
    assert learner.amp_order_trace == [("score", 0), ("vtrace", 0)]

    metrics = learner.update(batch)

    assert learner.discriminator_version == 1
    assert learner.amp_order_trace == [
        ("score", 0),
        ("vtrace", 0),
        ("policy", 0),
        ("discriminator", 1),
    ]
    assert metrics["amp/discriminator_version"] == 1.0
    assert metrics["amp/task_reward_mean"] == torch.mean(task_reward).item()
    assert metrics["amp/combined_reward_mean"] == torch.mean(expected).item()
    assert metrics["amp/expert_motion_count"] == 2.0
    assert metrics["amp/expert_transition_count"] == 935.0
    assert metrics["amp/expert_draw_count"] == 8.0
    assert 0.0 <= metrics["amp/policy_zero_style_fraction"] <= 1.0
    assert metrics["amp/task_weighted_mean"] == 0.75 * torch.mean(task_reward).item()


def test_amp_learner_checkpoint_roundtrip_is_exact(tmp_path: Path) -> None:
    learner = _learner(seed=11)
    batch = _batch()
    learner.process_batch(batch)
    learner.update(batch)
    checkpoint = tmp_path / "amp.pt"
    save_appo_checkpoint(checkpoint, learner.get_state_dict())

    restored = _learner(seed=999)
    restored.load_state_dict(load_appo_checkpoint(checkpoint))

    assert restored.discriminator_version == learner.discriminator_version
    assert restored.learning_rate == learner.learning_rate
    assert restored.optimizer.param_groups[0]["lr"] == learner.optimizer.param_groups[0]["lr"]
    assert restored.amp_replay.size == learner.amp_replay.size
    assert restored.amp_replay.write_position == learner.amp_replay.write_position
    for left, right in zip(
        learner.discriminator.state_dict().values(),
        restored.discriminator.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(left, right)
    for left, right in zip(
        learner.amp_normalizer.state_dict().values(),
        restored.amp_normalizer.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(learner.amp_replay.states, restored.amp_replay.states)
    torch.testing.assert_close(learner.amp_replay.next_states, restored.amp_replay.next_states)
    torch.testing.assert_close(
        learner.amp_generator.get_state(), restored.amp_generator.get_state()
    )
