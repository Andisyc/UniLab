"""AMP specialization of the generic APPO async runner."""

from __future__ import annotations

from typing import Any, cast

from unilab.algos.torch.amp.learner import AMPAPPOLearner
from unilab.algos.torch.amp.spec import AMP_OBSERVATION_DIM
from unilab.algos.torch.appo.learner import APPOLearner
from unilab.algos.torch.appo.runner import APPORunner
from unilab.ipc import RolloutFieldSpec


class AMPAPPORunner(APPORunner):
    def _learner_class(self):
        return AMPAPPOLearner

    def _learner_extra_kwargs(self) -> dict[str, Any]:
        amp_cfg = self.rl_cfg.get("amp")
        if not isinstance(amp_cfg, dict):
            raise ValueError("AMP APPO runtime requires algo.amp owner configuration")
        return {
            "amp_motion_manifest": amp_cfg["motion_manifest"],
            "amp_hidden_dims": amp_cfg.get("hidden_dims", [1024, 512, 256]),
            "amp_reward_coef": amp_cfg.get("reward_coef", 0.1),
            "amp_task_reward_lerp": amp_cfg.get("task_reward_lerp", 0.75),
            "amp_replay_capacity": amp_cfg.get("replay_capacity", 32768),
            "amp_discriminator_batch_size": amp_cfg.get("discriminator_batch_size", 4096),
            "amp_discriminator_updates": amp_cfg.get("discriminator_updates", 1),
            "amp_discriminator_learning_rate": amp_cfg.get("discriminator_learning_rate", 1e-3),
            "amp_gradient_penalty": amp_cfg.get("gradient_penalty", 10.0),
            "amp_seed": amp_cfg.get("seed", self.seed if self.seed is not None else 0),
        }

    def _extra_rollout_field_specs(self) -> dict[str, RolloutFieldSpec]:
        shape = (self.num_envs, self.steps_per_env, AMP_OBSERVATION_DIM)
        return {
            "amp_state": RolloutFieldSpec(shape=shape, dtype="float32", time_axis=True),
            "amp_next_state": RolloutFieldSpec(shape=shape, dtype="float32", time_axis=True),
        }

    def _collector_runtime_kwargs(self) -> dict[str, Any]:
        return {"rollout_payload_writer": "unilab.algos.torch.amp.worker:write_amp_rollout_payload"}

    def _restore_learner_checkpoint(self, learner: APPOLearner, checkpoint: dict[str, Any]) -> None:
        cast(AMPAPPOLearner, learner).load_state_dict(checkpoint)
