"""Learner-only AMP extension for UniLab's asynchronous APPO runtime."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from unilab.algos.torch.amp.motion_dataset import WalkMotionDataset
from unilab.algos.torch.amp.spec import AMP_OBSERVATION_DIM
from unilab.algos.torch.appo.learner import APPOLearner


class AMPDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], *, reward_coef: float) -> None:
        super().__init__()
        if input_dim != 2 * AMP_OBSERVATION_DIM:
            raise ValueError(f"AMP discriminator input_dim must be 390, got {input_dim}")
        if not hidden_dims:
            raise ValueError("AMP discriminator hidden_dims must be non-empty")
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, int(hidden)), nn.ReLU()))
            width = int(hidden)
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)
        self.reward_coef = float(reward_coef)

    def forward(self, transition: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(transition)).squeeze(-1)

    def predict_reward(
        self, state: torch.Tensor, next_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self(torch.cat((state, next_state), dim=-1))
            reward = self.reward_coef * torch.clamp(
                1.0 - 0.25 * torch.square(logits - 1.0), min=0.0
            )
        return reward, logits


class AMPRunningNormalizer(nn.Module):
    mean: torch.Tensor
    variance: torch.Tensor
    count: torch.Tensor

    def __init__(self, width: int, *, epsilon: float = 1e-4, clip: float = 10.0) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("variance", torch.ones(width))
        self.register_buffer("count", torch.zeros((), dtype=torch.float32))

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        normalized = (value - self.mean) / torch.sqrt(self.variance + self.epsilon)
        return torch.clamp(normalized, -self.clip, self.clip)

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        value = value.detach().reshape(-1, self.mean.numel()).to(self.mean.device)
        if value.shape[0] == 0:
            return
        batch_count = float(value.shape[0])
        batch_mean = value.mean(dim=0)
        batch_variance = value.var(dim=0, unbiased=False)
        old_count = float(self.count.item())
        if old_count == 0.0:
            self.mean.copy_(batch_mean)
            self.variance.copy_(batch_variance)
            self.count.fill_(batch_count)
            return
        total = old_count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / total)
        old_m2 = self.variance * old_count
        batch_m2 = batch_variance * batch_count
        correction = delta.square() * (old_count * batch_count / total)
        self.mean.copy_(new_mean)
        self.variance.copy_((old_m2 + batch_m2 + correction) / total)
        self.count.fill_(total)


class AMPReplayBuffer:
    def __init__(self, capacity: int, width: int, *, device: torch.device) -> None:
        if capacity < 1:
            raise ValueError("AMP replay capacity must be >= 1")
        self.capacity = int(capacity)
        self.width = int(width)
        self.device = device
        self.states = torch.zeros((capacity, width), device=device)
        self.next_states = torch.zeros((capacity, width), device=device)
        self.size = 0
        self.write_position = 0

    @torch.no_grad()
    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        states = states.detach().reshape(-1, self.width).to(self.device)
        next_states = next_states.detach().reshape(-1, self.width).to(self.device)
        if states.shape != next_states.shape:
            raise ValueError(
                f"AMP replay transition shape mismatch: {states.shape} vs {next_states.shape}"
            )
        if states.shape[0] >= self.capacity:
            self.states.copy_(states[-self.capacity :])
            self.next_states.copy_(next_states[-self.capacity :])
            self.size = self.capacity
            self.write_position = 0
            return
        count = states.shape[0]
        first = min(count, self.capacity - self.write_position)
        self.states[self.write_position : self.write_position + first].copy_(states[:first])
        self.next_states[self.write_position : self.write_position + first].copy_(
            next_states[:first]
        )
        remaining = count - first
        if remaining:
            self.states[:remaining].copy_(states[first:])
            self.next_states[:remaining].copy_(next_states[first:])
        self.write_position = (self.write_position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(
        self, count: int, *, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("cannot sample an empty AMP replay buffer")
        indices = torch.randint(self.size, (count,), generator=generator, device="cpu").to(
            self.device
        )
        return self.states[indices], self.next_states[indices]

    def state_dict(self) -> dict[str, Any]:
        return {
            "states": self.states,
            "next_states": self.next_states,
            "size": self.size,
            "write_position": self.write_position,
            "capacity": self.capacity,
            "width": self.width,
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity or int(state["width"]) != self.width:
            raise ValueError("AMP replay checkpoint shape does not match configured replay")
        self.states.copy_(state["states"].to(self.device))
        self.next_states.copy_(state["next_states"].to(self.device))
        self.size = int(state["size"])
        self.write_position = int(state["write_position"])


class AMPAPPOLearner(APPOLearner):
    def __init__(
        self,
        *args,
        amp_motion_manifest: str | Path,
        amp_hidden_dims: Sequence[int],
        amp_reward_coef: float = 0.1,
        amp_task_reward_lerp: float = 0.75,
        amp_replay_capacity: int = 32768,
        amp_discriminator_batch_size: int = 4096,
        amp_discriminator_updates: int = 1,
        amp_discriminator_learning_rate: float = 1e-3,
        amp_gradient_penalty: float = 10.0,
        amp_seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= amp_task_reward_lerp <= 1.0:
            raise ValueError("amp_task_reward_lerp must be in [0, 1]")
        if amp_discriminator_batch_size < 1 or amp_discriminator_updates < 1:
            raise ValueError("AMP discriminator batch size and updates must be >= 1")
        self.amp_task_reward_lerp = float(amp_task_reward_lerp)
        self.amp_discriminator_batch_size = int(amp_discriminator_batch_size)
        self.amp_discriminator_updates = int(amp_discriminator_updates)
        self.amp_gradient_penalty = float(amp_gradient_penalty)
        device = torch.device(self.device)
        self.discriminator = AMPDiscriminator(
            2 * AMP_OBSERVATION_DIM,
            amp_hidden_dims,
            reward_coef=amp_reward_coef,
        ).to(device)
        self.discriminator_optimizer = torch.optim.Adam(
            [
                {"params": self.discriminator.trunk.parameters(), "weight_decay": 1e-3},
                {"params": self.discriminator.head.parameters(), "weight_decay": 1e-1},
            ],
            lr=amp_discriminator_learning_rate,
        )
        self.amp_normalizer = AMPRunningNormalizer(AMP_OBSERVATION_DIM).to(device)
        self.amp_replay = AMPReplayBuffer(amp_replay_capacity, AMP_OBSERVATION_DIM, device=device)
        expert = WalkMotionDataset.from_manifest(amp_motion_manifest)
        self._expert_current = torch.from_numpy(expert.current_transitions).to(device)
        self._expert_next = torch.from_numpy(expert.next_transitions).to(device)
        self.amp_generator = torch.Generator(device="cpu").manual_seed(int(amp_seed))
        self.discriminator_version = 0
        self.amp_order_trace: list[tuple[str, int]] = []

    def predict_amp_reward(
        self, state: torch.Tensor, next_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = state.shape[:-1]
        state_flat = state.reshape(-1, AMP_OBSERVATION_DIM)
        next_flat = next_state.reshape(-1, AMP_OBSERVATION_DIM)
        normalized_state = self.amp_normalizer.normalize(state_flat)
        normalized_next = self.amp_normalizer.normalize(next_flat)
        reward, logits = self.discriminator.predict_reward(normalized_state, normalized_next)
        return reward.reshape(shape), logits.reshape(shape)

    def process_batch(self, batch_dict):
        if batch_dict.get("_amp_reward_applied", False):
            raise RuntimeError("AMP reward has already been applied to this staged batch")
        task_reward = batch_dict["rewards"].clone()
        amp_reward, logits = self.predict_amp_reward(
            batch_dict["amp_state"], batch_dict["amp_next_state"]
        )
        combined = (
            self.amp_task_reward_lerp * task_reward + (1.0 - self.amp_task_reward_lerp) * amp_reward
        )
        batch_dict["_amp_task_rewards"] = task_reward
        batch_dict["_amp_style_rewards"] = amp_reward
        batch_dict["_amp_logits"] = logits
        batch_dict["rewards"] = combined
        batch_dict["_amp_reward_applied"] = True
        self.amp_order_trace = [("score", self.discriminator_version)]
        self.amp_replay.insert(batch_dict["amp_state"], batch_dict["amp_next_state"])
        result = super().process_batch(batch_dict)
        self.amp_order_trace.append(("vtrace", self.discriminator_version))
        return result

    def update(self, batch_dict):
        metrics = super().update(batch_dict)
        self.amp_order_trace.append(("policy", self.discriminator_version))
        discriminator_metrics = self._update_amp_discriminator()
        self.discriminator_version += 1
        self.amp_order_trace.append(("discriminator", self.discriminator_version))
        metrics.update(discriminator_metrics)
        metrics.update(
            {
                "amp/discriminator_version": float(self.discriminator_version),
                "amp/task_reward_mean": float(batch_dict["_amp_task_rewards"].mean().item()),
                "amp/style_reward_mean": float(batch_dict["_amp_style_rewards"].mean().item()),
                "amp/combined_reward_mean": float(batch_dict["rewards"].mean().item()),
                "amp/replay_size": float(self.amp_replay.size),
            }
        )
        return metrics

    def _sample_expert(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(
            self._expert_current.shape[0],
            (count,),
            generator=self.amp_generator,
            device="cpu",
        ).to(self._expert_current.device)
        return self._expert_current[indices], self._expert_next[indices]

    def _update_amp_discriminator(self) -> dict[str, float]:
        loss_total = 0.0
        grad_penalty_total = 0.0
        policy_logit_total = 0.0
        expert_logit_total = 0.0
        normalizer_policy: torch.Tensor | None = None
        normalizer_expert: torch.Tensor | None = None
        for _ in range(self.amp_discriminator_updates):
            count = self.amp_discriminator_batch_size
            policy_state, policy_next = self.amp_replay.sample(count, generator=self.amp_generator)
            expert_state, expert_next = self._sample_expert(count)
            with torch.no_grad():
                normalized_policy_state = self.amp_normalizer.normalize(policy_state)
                normalized_policy_next = self.amp_normalizer.normalize(policy_next)
                normalized_expert_state = self.amp_normalizer.normalize(expert_state)
                normalized_expert_next = self.amp_normalizer.normalize(expert_next)
            policy_logits = self.discriminator(
                torch.cat((normalized_policy_state, normalized_policy_next), dim=-1)
            )
            expert_input = torch.cat(
                (normalized_expert_state, normalized_expert_next), dim=-1
            ).requires_grad_(True)
            expert_logits = self.discriminator(expert_input)
            expert_loss = torch.mean(torch.square(expert_logits - 1.0))
            policy_loss = torch.mean(torch.square(policy_logits + 1.0))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            gradient = torch.autograd.grad(
                expert_logits,
                expert_input,
                grad_outputs=torch.ones_like(expert_logits),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grad_penalty = self.amp_gradient_penalty * gradient.norm(2, dim=1).square().mean()
            loss = amp_loss + grad_penalty
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.discriminator_optimizer.step()
            loss_total += float(amp_loss.detach().item())
            grad_penalty_total += float(grad_penalty.detach().item())
            policy_logit_total += float(policy_logits.detach().mean().item())
            expert_logit_total += float(expert_logits.detach().mean().item())
            normalizer_policy = torch.cat((policy_state, policy_next), dim=0)
            normalizer_expert = torch.cat((expert_state, expert_next), dim=0)
        assert normalizer_policy is not None and normalizer_expert is not None
        self.amp_normalizer.update(torch.cat((normalizer_policy, normalizer_expert), dim=0))
        divisor = float(self.amp_discriminator_updates)
        return {
            "amp/discriminator_loss": loss_total / divisor,
            "amp/gradient_penalty": grad_penalty_total / divisor,
            "amp/policy_logit_mean": policy_logit_total / divisor,
            "amp/expert_logit_mean": expert_logit_total / divisor,
        }

    def get_state_dict(self):
        state = super().get_state_dict()
        state["amp"] = {
            "discriminator": self.discriminator.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "normalizer": self.amp_normalizer.state_dict(),
            "replay": self.amp_replay.state_dict(),
            "generator_state": self.amp_generator.get_state(),
            "discriminator_version": self.discriminator_version,
        }
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.optimizer.param_groups:
            self.learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self.target_actor.load_state_dict(self.actor.state_dict())
        amp = state.get("amp")
        if not isinstance(amp, dict):
            raise ValueError("AMP checkpoint is missing learner-local AMP state")
        self.discriminator.load_state_dict(amp["discriminator"])
        self.discriminator_optimizer.load_state_dict(amp["discriminator_optimizer"])
        self.amp_normalizer.load_state_dict(amp["normalizer"])
        self.amp_replay.load_state_dict(amp["replay"])
        self.amp_generator.set_state(amp["generator_state"].cpu())
        self.discriminator_version = int(amp["discriminator_version"])
