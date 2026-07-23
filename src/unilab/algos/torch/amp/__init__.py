"""Learner-owned adversarial motion-prior components."""

from unilab.algos.torch.amp.learner import AMPAPPOLearner, AMPDiscriminator
from unilab.algos.torch.amp.motion_dataset import AMPTransitionBatch, WalkMotionDataset
from unilab.algos.torch.amp.runner import AMPAPPORunner
from unilab.algos.torch.amp.spec import (
    AMP_ANCHOR_BODY_INDEX,
    AMP_ANCHOR_BODY_NAME,
    AMP_BODY_INDICES,
    AMP_BODY_NAMES,
    AMP_OBSERVATION_DIM,
    build_amp_observation,
    build_amp_observation_from_selected,
)

__all__ = [
    "AMPTransitionBatch",
    "AMPAPPOLearner",
    "AMPAPPORunner",
    "AMPDiscriminator",
    "WalkMotionDataset",
    "AMP_ANCHOR_BODY_INDEX",
    "AMP_ANCHOR_BODY_NAME",
    "AMP_BODY_INDICES",
    "AMP_BODY_NAMES",
    "AMP_OBSERVATION_DIM",
    "build_amp_observation",
    "build_amp_observation_from_selected",
]
