from __future__ import annotations

import ast
import builtins
import json
import types

import numpy as np
import pytest
import torch


def test_behavior_distillation_update_detaches_teacher_and_updates_student() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MLPStudentPolicy,
    )

    torch.manual_seed(7)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    teacher = torch.nn.Linear(7, 3)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
    )
    batch = DistillationBatch(
        student_obs=torch.randn(4, 5),
        teacher_obs=torch.randn(4, 7),
    )
    before = {name: value.detach().clone() for name, value in student.state_dict().items()}

    stats = trainer.update(batch)

    assert stats.update_count == 1
    assert stats.loss > 0.0
    assert stats.student_grad_norm > 0.0
    assert stats.teacher_action_requires_grad is False
    assert stats.student_action_shape == (4, 3)
    assert stats.teacher_action_shape == (4, 3)
    assert all(param.grad is None for param in teacher.parameters())
    assert any(
        not torch.allclose(before[name], value) for name, value in student.state_dict().items()
    )


def test_command_intent_rollout_resolves_deployment_experts() -> None:
    from unilab.algos.torch.distill import (
        MoEStudentPolicy,
        resolve_command_intent_rollout_policies,
    )

    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=3,
        num_experts=3,
        expert_hidden_dims=(4,),
    )
    policies, targets = resolve_command_intent_rollout_policies(
        student,
        {"command_intent_expert_targets": {"active": 0, "inactive": 1}},
    )

    assert targets == {"active": 0, "inactive": 1}
    assert policies["active"] is student.experts[0]
    assert policies["inactive"] is student.experts[1]


def test_command_intent_rollout_rejects_missing_target() -> None:
    from unilab.algos.torch.distill import MoEStudentPolicy, resolve_command_intent_rollout_policies

    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=3,
        num_experts=3,
        expert_hidden_dims=(4,),
    )
    with pytest.raises(ValueError, match="missing intents"):
        resolve_command_intent_rollout_policies(
            student,
            {"command_intent_expert_targets": {"active": 0}},
        )


def test_behavior_distillation_update_uses_cached_teacher_actions() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MLPStudentPolicy,
    )

    class RaisingTeacher(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            del obs
            raise AssertionError("cached teacher_action path must not call teacher")

    torch.manual_seed(13)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=RaisingTeacher(),
        optimizer=optimizer,
    )
    batch = DistillationBatch(
        student_obs=torch.randn(4, 5),
        teacher_obs=torch.empty(4, 0),
        teacher_actions=torch.randn(4, 3, requires_grad=True),
    )

    stats = trainer.update(batch)

    assert stats.update_count == 1
    assert stats.loss > 0.0
    assert stats.student_grad_norm > 0.0
    assert stats.teacher_action_shape == (4, 3)
    assert stats.teacher_action_requires_grad is False
    assert stats.teacher_action_source == "cached"


def test_behavior_distillation_checkpoint_roundtrip(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        MLPStudentPolicy,
        load_distillation_checkpoint,
        save_distillation_checkpoint,
    )

    torch.manual_seed(11)
    source = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    target = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    checkpoint_path = tmp_path / "nested" / "distill_model.pt"

    save_distillation_checkpoint(
        checkpoint_path,
        student=source,
        agent_steps=16,
        teacher_metadata={"algo": "sac", "task": "G1WalkHeight"},
        distill_runtime_cfg={"loss_type": "mse"},
    )
    checkpoint = load_distillation_checkpoint(target, checkpoint_path)

    assert checkpoint_path.is_file()
    assert checkpoint["agent_steps"] == 16
    assert checkpoint["teacher_metadata"] == {"algo": "sac", "task": "G1WalkHeight"}
    assert checkpoint["distill_runtime_cfg"] == {"loss_type": "mse"}
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        assert torch.allclose(source_param, target_param)


def test_offline_distillation_checkpoint_can_omit_optimizer_state(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        run_offline_distillation_updates,
    )

    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    teacher = torch.nn.Linear(7, 3)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
    )
    dataset = build_distillation_dataset(
        torch.randn(4, 5),
        torch.randn(4, 7),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
        expected_teacher_action_dim=3,
        teacher_actions=torch.randn(4, 3),
    )
    checkpoint_path = tmp_path / "student_no_optimizer.pt"

    run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=2,
        max_updates=1,
        checkpoint_path=checkpoint_path,
        save_optimizer_state=False,
    )

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "student_state_dict" in raw
    assert "optimizer_state_dict" not in raw
    assert not list(tmp_path.glob(".student_no_optimizer.pt.tmp.*"))


def test_behavior_distillation_rejects_batch_shape_mismatch() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MLPStudentPolicy,
    )

    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    teacher = torch.nn.Linear(7, 3)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
    )

    try:
        trainer.update(
            DistillationBatch(
                student_obs=torch.randn(4, 5),
                teacher_obs=torch.randn(5, 7),
            )
        )
    except ValueError as exc:
        assert "batch size" in str(exc)
    else:
        raise AssertionError("expected shape mismatch to raise ValueError")


def test_moe_student_policy_routes_and_mixes_expert_actions() -> None:
    from unilab.algos.torch.distill import MoEStudentPolicy

    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=2,
        num_experts=3,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        routing_mode="soft",
        squash_action=False,
    )
    with torch.no_grad():
        for expert, bias in zip(
            student.experts,
            (
                torch.tensor([1.0, 0.0]),
                torch.tensor([0.0, 2.0]),
                torch.tensor([-1.0, 1.0]),
            ),
            strict=True,
        ):
            expert.net[-1].weight.zero_()
            expert.net[-1].bias.copy_(bias)
        student.router[-1].weight.zero_()
        student.router[-1].bias.zero_()

    obs = torch.zeros(2, 4)
    output = student(obs, return_diagnostics=True)

    assert output.action.shape == (2, 2)
    assert output.router_logits.shape == (2, 3)
    assert output.route_probs.shape == (2, 3)
    assert output.expert_actions.shape == (2, 3, 2)
    assert output.selected_expert is None
    assert torch.allclose(output.route_probs, torch.full((2, 3), 1.0 / 3.0))
    assert torch.allclose(output.expert_usage, torch.full((3,), 2.0 / 3.0))
    assert torch.allclose(output.action, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))

    with torch.no_grad():
        student.router[-1].bias.copy_(torch.tensor([-2.0, 3.0, -1.0]))
    hard_output = student(obs, hard_routing=True, return_diagnostics=True)

    assert torch.equal(hard_output.selected_expert, torch.ones(2, dtype=torch.long))
    assert torch.allclose(hard_output.expert_usage, torch.tensor([0.0, 2.0, 0.0]))
    assert torch.allclose(hard_output.action, torch.tensor([[0.0, 2.0], [0.0, 2.0]]))


def test_moe_student_policy_soft_route_backpropagates_router_and_experts() -> None:
    from unilab.algos.torch.distill import MoEStudentPolicy

    torch.manual_seed(41)
    student = MoEStudentPolicy(
        obs_dim=5,
        action_dim=3,
        num_experts=2,
        expert_hidden_dims=(8,),
        router_hidden_dims=(4,),
    )
    output = student(torch.randn(4, 5), return_diagnostics=True)
    loss = output.action.pow(2).mean() + output.route_probs[:, 0].mean()

    loss.backward()

    router_grad_norm = sum(
        float(param.grad.detach().pow(2).sum().item())
        for param in student.router.parameters()
        if param.grad is not None
    )
    expert_grad_norm = sum(
        float(param.grad.detach().pow(2).sum().item())
        for expert in student.experts
        for param in expert.parameters()
        if param.grad is not None
    )
    assert output.action.shape == (4, 3)
    assert output.action.requires_grad is True
    assert router_grad_norm > 0.0
    assert expert_grad_norm > 0.0


def test_moe_student_policy_rejects_bad_contract() -> None:
    from unilab.algos.torch.distill import MoEStudentPolicy

    with pytest.raises(ValueError, match="num_experts"):
        MoEStudentPolicy(obs_dim=4, action_dim=2, num_experts=1)

    with pytest.raises(ValueError, match="router_temperature"):
        MoEStudentPolicy(obs_dim=4, action_dim=2, num_experts=2, router_temperature=0.0)

    student = MoEStudentPolicy(obs_dim=4, action_dim=2, num_experts=2)
    with pytest.raises(ValueError, match="Student obs dim mismatch"):
        student(torch.zeros(3, 5))


def test_moe_distillation_trainer_records_aux_loss_and_usage() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MoEStudentPolicy,
    )

    torch.manual_seed(43)
    student = MoEStudentPolicy(
        obs_dim=5,
        action_dim=3,
        num_experts=2,
        expert_hidden_dims=(8,),
        router_hidden_dims=(4,),
    )
    teacher = torch.nn.Linear(7, 3)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        aux_loss_coef=0.25,
    )
    stats = trainer.update(
        DistillationBatch(
            student_obs=torch.randn(4, 5),
            teacher_obs=torch.randn(4, 7),
        )
    )

    router_grad_norm = sum(
        float(param.grad.detach().pow(2).sum().item())
        for param in student.router.parameters()
        if param.grad is not None
    )
    expert_grad_norm = sum(
        float(param.grad.detach().pow(2).sum().item())
        for expert in student.experts
        for param in expert.parameters()
        if param.grad is not None
    )
    assert stats.behavior_loss > 0.0
    assert stats.aux_loss >= 0.0
    assert stats.loss == pytest.approx(stats.behavior_loss + 0.25 * stats.aux_loss)
    assert stats.student_action_shape == (4, 3)
    assert stats.teacher_action_shape == (4, 3)
    assert stats.expert_usage is not None
    assert len(stats.expert_usage) == 2
    assert sum(stats.expert_usage) == pytest.approx(4.0)
    assert stats.route_entropy is not None
    assert stats.route_entropy >= 0.0
    assert router_grad_norm > 0.0
    assert expert_grad_norm > 0.0


def test_moe_distillation_trainer_applies_role_conditioned_router_loss() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MoEStudentPolicy,
    )

    torch.manual_seed(47)
    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=2,
        num_experts=3,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        squash_action=False,
    )
    with torch.no_grad():
        for expert in student.experts:
            expert.net[-1].weight.zero_()
            expert.net[-1].bias.zero_()
        student.router[-1].weight.zero_()
        student.router[-1].bias.zero_()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Identity(),
        optimizer=optimizer,
        role_loss_coef=0.5,
        role_expert_targets={"stand": 0, "walk": 1, "height": 2},
    )

    stats = trainer.update(
        DistillationBatch(
            student_obs=torch.eye(4),
            teacher_obs=torch.empty(4, 0),
            role_labels=("stand", "stand", "walk", "height"),
            teacher_actions=torch.zeros(4, 2),
        )
    )

    router_grad_norm = sum(
        float(param.grad.detach().pow(2).sum().item())
        for param in student.router.parameters()
        if param.grad is not None
    )
    assert stats.behavior_loss == pytest.approx(0.0)
    assert stats.aux_loss == pytest.approx(0.0)
    assert stats.role_loss > 0.0
    assert stats.role_target_count == 4
    assert stats.loss == pytest.approx(0.5 * stats.role_loss)
    assert router_grad_norm > 0.0


def test_moe_distillation_trainer_applies_command_intent_router_loss() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MoEStudentPolicy,
    )

    torch.manual_seed(47)
    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=2,
        num_experts=2,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        squash_action=False,
    )
    with torch.no_grad():
        for expert in student.experts:
            expert.net[-1].weight.zero_()
            expert.net[-1].bias.zero_()
        student.router[-1].weight.zero_()
        student.router[-1].bias.zero_()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Identity(),
        optimizer=optimizer,
        command_intent_loss_coef=0.75,
        command_intent_expert_targets={"inactive": 0, "active": 1},
    )

    stats = trainer.update(
        DistillationBatch(
            student_obs=torch.eye(4),
            teacher_obs=torch.empty(4, 0),
            command_intents=("inactive", "inactive", "active", "active"),
            teacher_actions=torch.zeros(4, 2),
        )
    )

    router_grad_norm = sum(
        float(param.grad.detach().pow(2).sum().item())
        for param in student.router.parameters()
        if param.grad is not None
    )
    assert stats.behavior_loss == pytest.approx(0.0)
    assert stats.aux_loss == pytest.approx(0.0)
    assert stats.role_loss == pytest.approx(0.0)
    assert stats.command_intent_loss > 0.0
    assert stats.command_intent_target_count == 4
    assert stats.loss == pytest.approx(0.75 * stats.command_intent_loss)
    assert router_grad_norm > 0.0


def test_distillation_runtime_trace_covers_command_target_chain(monkeypatch, capsys) -> None:
    monkeypatch.setenv("UNILAB_DISTILL_RUNTIME_DEBUG", "1")
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MoEStudentPolicy,
    )

    student = MoEStudentPolicy(
        obs_dim=2,
        action_dim=2,
        num_experts=2,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        squash_action=False,
    )
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Identity(),
        optimizer=torch.optim.Adam(student.parameters(), lr=0.0),
        command_intent_loss_coef=1.0,
        command_intent_expert_targets={"inactive": 0, "active": 1},
        expert_behavior_loss_source="none",
    )

    trainer.update(
        DistillationBatch(
            student_obs=torch.zeros(2, 2),
            teacher_obs=torch.empty(2, 0),
            command_intents=("inactive", "active"),
            teacher_actions=torch.zeros(2, 2),
        )
    )

    prefix = "[distill-trainer-runtime] "
    snapshots = [
        ast.literal_eval(line.removeprefix(prefix))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(prefix)
    ]
    stages = [snapshot["stage"] for snapshot in snapshots]
    assert stages == [
        "trainer/update_entry",
        "trainer/after_student_forward",
        "trainer/after_role_loss",
        "command_intent/entry",
        "target_index/before_int",
        "target_index/after_int",
        "target_index/after_append",
        "target_index/before_int",
        "target_index/after_int",
        "target_index/after_append",
        "command_intent/after_target_tensor",
        "trainer/after_command_intent_loss",
        "trainer/after_behavior_action",
        "trainer/after_loss",
        "trainer/before_backward",
        "trainer/after_backward",
        "trainer/before_optimizer",
        "trainer/after_optimizer",
        "trainer/update_complete",
    ]
    target_snapshots = [
        snapshot for snapshot in snapshots if snapshot["stage"] == "target_index/before_int"
    ]
    assert [snapshot["label_key"] for snapshot in target_snapshots] == [
        "inactive",
        "active",
    ]
    assert all(snapshot["builtins_int_callable"] is True for snapshot in target_snapshots)
    assert all(snapshot["append_callable"] is True for snapshot in target_snapshots)


def test_distillation_runtime_trace_identifies_int_callable_corruption(monkeypatch, capsys) -> None:
    monkeypatch.setenv("UNILAB_DISTILL_RUNTIME_DEBUG", "1")
    from unilab.algos.torch.distill import BehaviorDistillationTrainer, MoEStudentPolicy

    student = MoEStudentPolicy(
        obs_dim=2,
        action_dim=2,
        num_experts=2,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        squash_action=False,
    )
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Identity(),
        optimizer=torch.optim.Adam(student.parameters(), lr=0.0),
        command_intent_loss_coef=1.0,
        command_intent_expert_targets={"active": 1},
    )
    original_int = builtins.int

    class MutatingIntent:
        def __str__(self) -> str:
            builtins.int = iter((1,))  # type: ignore[assignment]
            return "active"

    error = None
    try:
        trainer._command_intent_router_loss(
            command_intents=(MutatingIntent(),),  # type: ignore[arg-type]
            router_logits=torch.zeros(1, 2),
            batch_size=1,
            like=torch.zeros(()),
        )
    except TypeError as caught:
        error = caught
    finally:
        builtins.int = original_int  # type: ignore[assignment]

    assert error is not None
    assert "tuple_iterator" in str(error)
    prefix = "[distill-trainer-runtime] "
    snapshots = [
        ast.literal_eval(line.removeprefix(prefix))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(prefix)
    ]
    failure = snapshots[-1]
    assert failure["stage"] == "target_index/int_failure"
    assert failure["label_name"] == "command_intent"
    assert failure["label_key"] == "active"
    assert failure["row_index"] == 0
    assert failure["builtins_int_type"] == "tuple_iterator"
    assert failure["builtins_int_callable"] is False
    assert failure["target_indices_type"] == "list"
    assert failure["append_callable"] is True
    assert failure["raw_target_repr"] == "1"


def test_distillation_runtime_trace_identifies_target_tensor_list_corruption(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("UNILAB_DISTILL_RUNTIME_DEBUG", "1")
    from unilab.algos.torch.distill import BehaviorDistillationTrainer, MoEStudentPolicy

    student = MoEStudentPolicy(
        obs_dim=2,
        action_dim=2,
        num_experts=2,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        squash_action=False,
    )
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Identity(),
        optimizer=torch.optim.Adam(student.parameters(), lr=0.0),
        command_intent_expert_targets={"inactive": 0, "active": 1},
    )

    def append_corrupted_target(**kwargs) -> None:
        target_indices = kwargs["target_indices"]
        target_indices.append(0 if kwargs["row_index"] == 0 else None)

    monkeypatch.setattr(
        trainer,
        "_append_runtime_target_index",
        append_corrupted_target,
    )
    with pytest.raises(TypeError, match="NoneType.*interpreted as an integer"):
        trainer._target_indices_from_labels(
            labels=("inactive", "active"),
            targets={"inactive": 0, "active": 1},
            batch_size=2,
            num_experts=2,
            label_name="command_intent",
            required=True,
            device=torch.device("cpu"),
        )

    prefix = "[distill-trainer-runtime] "
    snapshots = [
        ast.literal_eval(line.removeprefix(prefix))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(prefix)
    ]
    assert len(snapshots) == 1
    failure = snapshots[0]
    assert failure["stage"] == "target_index/tensor_failure"
    assert failure["label_name"] == "command_intent"
    assert failure["update_number"] == 1
    assert failure["batch_size"] == 2
    assert failure["num_experts"] == 2
    assert failure["target_indices"]["length"] == 2
    assert failure["target_indices"]["none_count"] == 1
    assert failure["target_indices"]["non_int_count"] == 1
    assert failure["target_indices"]["element_type_counts"] == {
        "NoneType": 1,
        "int": 1,
    }
    assert failure["target_indices"]["invalid_head"] == [
        {"index": 1, "raw_repr": "None", "raw_type": "NoneType"}
    ]
    assert failure["torch_tensor_callable"] is True
    assert failure["torch_tensor_is_original"] is True
    assert failure["error_type"] == "TypeError"


def test_moe_distillation_trainer_uses_command_intent_expert_behavior_loss() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MoEStudentPolicy,
    )

    student = MoEStudentPolicy(
        obs_dim=2,
        action_dim=2,
        num_experts=2,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        squash_action=False,
    )
    with torch.no_grad():
        student.router[-1].weight.zero_()
        student.router[-1].bias.zero_()
        for expert, bias in zip(
            student.experts,
            (torch.tensor([0.0, 0.0]), torch.tensor([10.0, 10.0])),
            strict=True,
        ):
            expert.net[-1].weight.zero_()
            expert.net[-1].bias.copy_(bias)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.0)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Identity(),
        optimizer=optimizer,
        command_intent_expert_targets={"inactive": 0, "active": 1},
    )

    stats = trainer.update(
        DistillationBatch(
            student_obs=torch.zeros(2, 2),
            teacher_obs=torch.empty(2, 0),
            command_intents=("inactive", "active"),
            teacher_actions=torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
        )
    )

    assert stats.behavior_loss == pytest.approx(0.0)
    assert stats.behavior_action_shape == (2, 2)
    assert stats.behavior_action_source == "command_intent_expert"
    assert stats.behavior_target_count == 2
    assert stats.loss == pytest.approx(0.0)


def test_moe_expert_behavior_loss_rejects_conflicting_role_and_intent_targets() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MoEStudentPolicy,
    )

    student = MoEStudentPolicy(
        obs_dim=2,
        action_dim=2,
        num_experts=2,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        squash_action=False,
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=0.0)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Identity(),
        optimizer=optimizer,
        role_expert_targets={"stand": 0},
        command_intent_expert_targets={"inactive": 1},
    )

    with pytest.raises(ValueError, match="conflict"):
        trainer.update(
            DistillationBatch(
                student_obs=torch.zeros(1, 2),
                teacher_obs=torch.empty(1, 0),
                role_labels=("stand",),
                command_intents=("inactive",),
                teacher_actions=torch.zeros(1, 2),
            )
        )


def test_moe_role_conditioned_router_loss_fails_closed() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MLPStudentPolicy,
        MoEStudentPolicy,
    )

    student = MoEStudentPolicy(obs_dim=4, action_dim=2, num_experts=2)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    with pytest.raises(ValueError, match="role_expert_targets"):
        BehaviorDistillationTrainer(
            student=student,
            teacher=torch.nn.Identity(),
            optimizer=optimizer,
            role_loss_coef=0.1,
        )

    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Identity(),
        optimizer=optimizer,
        role_loss_coef=0.1,
        role_expert_targets={"stand": 0},
    )
    with pytest.raises(ValueError, match="role_labels"):
        trainer.update(
            DistillationBatch(
                student_obs=torch.zeros(2, 4),
                teacher_obs=torch.empty(2, 0),
                teacher_actions=torch.zeros(2, 2),
            )
        )
    with pytest.raises(ValueError, match="unmapped role label"):
        trainer.update(
            DistillationBatch(
                student_obs=torch.zeros(2, 4),
                teacher_obs=torch.empty(2, 0),
                role_labels=("stand", "walk"),
                teacher_actions=torch.zeros(2, 2),
            )
        )

    mlp = MLPStudentPolicy(obs_dim=4, action_dim=2, hidden_dims=(8,))
    mlp_trainer = BehaviorDistillationTrainer(
        student=mlp,
        teacher=torch.nn.Identity(),
        optimizer=torch.optim.Adam(mlp.parameters(), lr=1e-2),
        role_loss_coef=0.1,
        role_expert_targets={"stand": 0},
    )
    with pytest.raises(TypeError, match="router logits"):
        mlp_trainer.update(
            DistillationBatch(
                student_obs=torch.zeros(2, 4),
                teacher_obs=torch.empty(2, 0),
                role_labels=("stand", "stand"),
                teacher_actions=torch.zeros(2, 2),
            )
        )


def test_moe_command_intent_router_loss_fails_closed() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MLPStudentPolicy,
        MoEStudentPolicy,
    )

    student = MoEStudentPolicy(obs_dim=4, action_dim=2, num_experts=2)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    with pytest.raises(ValueError, match="command_intent_expert_targets"):
        BehaviorDistillationTrainer(
            student=student,
            teacher=torch.nn.Identity(),
            optimizer=optimizer,
            command_intent_loss_coef=0.1,
        )

    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Identity(),
        optimizer=optimizer,
        command_intent_loss_coef=0.1,
        command_intent_expert_targets={"inactive": 0},
    )
    with pytest.raises(ValueError, match="command_intents"):
        trainer.update(
            DistillationBatch(
                student_obs=torch.zeros(2, 4),
                teacher_obs=torch.empty(2, 0),
                teacher_actions=torch.zeros(2, 2),
            )
        )
    with pytest.raises(ValueError, match="unmapped command intent"):
        trainer.update(
            DistillationBatch(
                student_obs=torch.zeros(2, 4),
                teacher_obs=torch.empty(2, 0),
                command_intents=("inactive", "active"),
                teacher_actions=torch.zeros(2, 2),
            )
        )

    mlp = MLPStudentPolicy(obs_dim=4, action_dim=2, hidden_dims=(8,))
    mlp_trainer = BehaviorDistillationTrainer(
        student=mlp,
        teacher=torch.nn.Identity(),
        optimizer=torch.optim.Adam(mlp.parameters(), lr=1e-2),
        command_intent_loss_coef=0.1,
        command_intent_expert_targets={"inactive": 0},
    )
    with pytest.raises(TypeError, match="router logits"):
        mlp_trainer.update(
            DistillationBatch(
                student_obs=torch.zeros(2, 4),
                teacher_obs=torch.empty(2, 0),
                command_intents=("inactive", "inactive"),
                teacher_actions=torch.zeros(2, 2),
            )
        )


def test_moe_expert_diagnostics_explain_toy_roles() -> None:
    from unilab.algos.torch.distill import (
        MoEStudentPolicy,
        diagnose_moe_expert_routes,
        moe_diagnostics_to_dict,
    )

    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=2,
        num_experts=3,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        routing_mode="hard",
        squash_action=False,
    )
    with torch.no_grad():
        student.router[-1].weight.zero_()
        student.router[-1].bias.zero_()
        student.router[-1].weight[0, 0] = 4.0
        student.router[-1].weight[1, 1] = 4.0
        student.router[-1].weight[2, 2] = 4.0

    obs = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 1.5, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 1.5, 0.0],
        ],
        dtype=torch.float32,
    )
    diagnostics = diagnose_moe_expert_routes(
        student,
        obs,
        role_labels=["stand", "stand", "walk", "walk", "recovery", "recovery"],
        hard_routing=True,
        collapse_fraction=0.95,
    )
    by_role = {summary.role: summary for summary in diagnostics.by_role}
    payload = moe_diagnostics_to_dict(diagnostics)

    assert diagnostics.role_labels_present is True
    assert diagnostics.num_samples == 6
    assert diagnostics.num_experts == 3
    assert diagnostics.overall.expert_usage == pytest.approx((2.0, 2.0, 2.0))
    assert diagnostics.overall.collapse_detected is False
    assert by_role["stand"].dominant_expert == 0
    assert by_role["walk"].dominant_expert == 1
    assert by_role["recovery"].dominant_expert == 2
    assert by_role["stand"].expert_fraction == pytest.approx((1.0, 0.0, 0.0))
    assert by_role["walk"].expert_fraction == pytest.approx((0.0, 1.0, 0.0))
    assert by_role["recovery"].expert_fraction == pytest.approx((0.0, 0.0, 1.0))
    assert payload["by_role"][0]["role"] == "recovery"
    assert payload["overall"]["collapse_detected"] is False


def test_moe_expert_diagnostics_flags_router_collapse_and_label_errors() -> None:
    from unilab.algos.torch.distill import MoEStudentPolicy, diagnose_moe_expert_routes

    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=2,
        num_experts=3,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        routing_mode="hard",
        squash_action=False,
    )
    with torch.no_grad():
        student.router[-1].weight.zero_()
        student.router[-1].bias.copy_(torch.tensor([5.0, 0.0, 0.0]))

    obs = torch.zeros(4, 4)
    diagnostics = diagnose_moe_expert_routes(
        student,
        obs,
        role_labels=["stand", "stand", "stand", "stand"],
        hard_routing=True,
        collapse_fraction=0.75,
    )

    assert diagnostics.overall.dominant_expert == 0
    assert diagnostics.overall.expert_fraction == pytest.approx((1.0, 0.0, 0.0))
    assert diagnostics.overall.collapse_detected is True
    assert diagnostics.by_role[0].collapse_detected is True

    with pytest.raises(ValueError, match="role_labels length"):
        diagnose_moe_expert_routes(student, obs, role_labels=["stand"])


def test_moe_expert_semantics_probe_reports_cached_action_error(tmp_path) -> None:
    from scripts.deploy.check_unilab_g1_distill_moe_expert_semantics import run_check

    from unilab.algos.torch.distill import (
        MoEStudentPolicy,
        build_distillation_dataset,
        save_distillation_checkpoint,
        save_distillation_dataset,
    )

    student = MoEStudentPolicy(
        obs_dim=2,
        action_dim=2,
        num_experts=2,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        routing_mode="hard",
        squash_action=False,
    )
    with torch.no_grad():
        for expert, bias in zip(
            student.experts,
            (torch.tensor([0.1, -0.1]), torch.tensor([0.5, 0.2])),
            strict=True,
        ):
            expert.net[-1].weight.zero_()
            expert.net[-1].bias.copy_(bias)
        student.router[-1].weight.zero_()
        student.router[-1].bias.zero_()
        student.router[-1].weight[0, 0] = 4.0
        student.router[-1].weight[1, 1] = 4.0

    student_obs = torch.tensor(
        [[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    teacher_actions = torch.tensor(
        [[0.1, -0.1], [0.1, -0.1], [0.5, 0.2], [0.5, 0.2]],
        dtype=torch.float32,
    )
    dataset_path = tmp_path / "role_dataset.pt"
    checkpoint_path = tmp_path / "moe_student.pt"
    dataset = build_distillation_dataset(
        student_obs,
        torch.empty(4, 0),
        expected_student_obs_dim=2,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=2,
        teacher_actions=teacher_actions,
        role_labels=("stand", "stand", "walk_flat", "walk_flat"),
    )
    save_distillation_dataset(dataset_path, dataset)
    save_distillation_checkpoint(
        checkpoint_path,
        student=student,
        agent_steps=4,
        distill_runtime_cfg={
            "student_model_type": "moe",
            "student_obs_dim": 2,
            "teacher_obs_dim": 0,
            "student_action_dim": 2,
            "student_num_experts": 2,
            "student_expert_hidden_dims": [],
            "student_router_hidden_dims": [],
            "student_routing_mode": "hard",
            "student_squash_action": False,
        },
    )

    checks, details = run_check(
        task="g1_walk_flat/mujoco",
        dataset_path=dataset_path,
        student_checkpoint=checkpoint_path,
        hard_routing=True,
    )

    action_imitation = details["moe_expert/action_imitation"]
    dataset_metadata = details["moe_expert/dataset_metadata"]
    assert all(check.level != "FAIL" for check in checks)
    assert "role_labels" not in dataset_metadata
    assert dataset_metadata["role_label_counts"] == {"stand": 2, "walk_flat": 2}
    assert dataset_metadata["role_label_count_total"] == 4
    assert action_imitation["overall"]["mse"] == pytest.approx(0.0)
    assert action_imitation["by_role"]["stand"]["mse"] == pytest.approx(0.0)
    assert action_imitation["by_role"]["walk_flat"]["mse"] == pytest.approx(0.0)
    assert action_imitation["by_role"]["stand"]["student_action_abs_max"] == pytest.approx(0.1)
    assert action_imitation["by_role"]["walk_flat"]["student_action_abs_max"] == pytest.approx(0.5)


def test_distillation_dataset_roundtrip_preserves_obs_batch_contract(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        load_distillation_dataset,
        save_distillation_dataset,
    )

    student_obs = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    teacher_obs = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    dataset = build_distillation_dataset(
        student_obs,
        teacher_obs,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
        metadata={"source": "offline-fixture"},
    )

    batch = dataset.as_batch(start=1, batch_size=2)
    assert batch.student_obs.shape == (2, 5)
    assert batch.teacher_obs.shape == (2, 7)
    assert torch.equal(batch.student_obs[0], student_obs[1])
    assert torch.equal(batch.teacher_obs[1], teacher_obs[2])

    checkpoint_path = tmp_path / "distill_dataset.pt"
    save_distillation_dataset(checkpoint_path, dataset)
    restored = load_distillation_dataset(
        checkpoint_path,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
    )

    assert restored.num_samples == 4
    assert restored.student_obs_dim == 5
    assert restored.teacher_obs_dim == 7
    assert restored.metadata["source"] == "offline-fixture"
    assert torch.equal(restored.student_obs, student_obs)
    assert torch.equal(restored.teacher_obs, teacher_obs)
    assert restored.role_labels is None


def test_distillation_dataset_to_moves_all_tensors_and_preserves_labels() -> None:
    from unilab.algos.torch.distill import build_distillation_dataset

    dataset = build_distillation_dataset(
        torch.zeros(2, 8),
        torch.ones(2, 8),
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        expected_teacher_action_dim=3,
        metadata={"source": "device-contract"},
        role_labels=("stand", "walk_flat"),
        teacher_actions=torch.full((2, 3), 0.25),
        commands=torch.zeros(2, 3),
        command_intents=("inactive", "active"),
    )

    moved = dataset.to("meta")

    assert moved.student_obs.device.type == "meta"
    assert moved.teacher_obs.device.type == "meta"
    assert moved.teacher_actions is not None
    assert moved.teacher_actions.device.type == "meta"
    assert moved.commands is not None
    assert moved.commands.device.type == "meta"
    assert moved.role_labels == dataset.role_labels
    assert moved.command_intents == dataset.command_intents
    assert moved.metadata == dataset.metadata


def test_distillation_dataset_roundtrip_preserves_role_labels_contract(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        load_distillation_dataset,
        save_distillation_dataset,
    )

    student_obs = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    teacher_obs = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    role_labels = ("stand", "walk_height", "stand_height", "walk_height")
    dataset = build_distillation_dataset(
        student_obs,
        teacher_obs,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
        metadata={"source": "role-fixture"},
        role_labels=role_labels,
    )

    batch = dataset.as_batch(start=1, batch_size=2)
    assert batch.role_labels == ("walk_height", "stand_height")

    checkpoint_path = tmp_path / "role_distill_dataset.pt"
    save_distillation_dataset(checkpoint_path, dataset)
    restored = load_distillation_dataset(
        checkpoint_path,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
    )

    assert restored.role_labels == role_labels
    assert restored.metadata["role_labels"] == list(role_labels)
    assert restored.command_intents == ("inactive", "active", "inactive", "active")
    assert restored.metadata["command_intent_inference_source"] == "role_labels"
    assert restored.metadata["command_intent_counts"] == {"active": 2, "inactive": 2}
    assert restored.as_batch(start=2, batch_size=8).role_labels == (
        "stand_height",
        "walk_height",
    )


def test_distillation_dataset_infers_command_intents_from_legacy_roles(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        load_distillation_dataset,
        run_offline_distillation_updates,
    )

    checkpoint_path = tmp_path / "legacy_role_dataset.pt"
    torch.save(
        {
            "student_obs": torch.randn(4, 5),
            "teacher_obs": torch.empty(4, 0),
            "teacher_actions": torch.randn(4, 3),
            "metadata": {"source": "legacy-role-only"},
            "role_labels": ["walk_flat", "stand", "g1_walk_flat", "g1_stand_still"],
            "num_samples": 4,
        },
        checkpoint_path,
    )
    restored = load_distillation_dataset(
        checkpoint_path,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
    )

    assert restored.command_intents == ("active", "inactive", "active", "inactive")
    assert restored.metadata["command_intent_inference_source"] == "role_labels"
    assert restored.metadata["command_intent_counts"] == {"active": 2, "inactive": 2}

    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Linear(0, 3),
        optimizer=torch.optim.Adam(student.parameters(), lr=1e-2),
    )
    result = run_offline_distillation_updates(
        trainer,
        restored,
        batch_size=4,
        max_updates=1,
        balance_key="command_intent",
        balanced_labels=("inactive", "active"),
    )

    assert result.last_balance_label_counts == {"inactive": 2, "active": 2}


def test_distillation_dataset_keeps_unknown_roles_without_intent_guess() -> None:
    from unilab.algos.torch.distill import build_distillation_dataset

    dataset = build_distillation_dataset(
        torch.randn(2, 5),
        torch.empty(2, 0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
        teacher_actions=torch.randn(2, 3),
        role_labels=("height_low", "height_high"),
    )

    assert dataset.command_intents is None
    assert "command_intent_inference_source" not in dataset.metadata


def test_distillation_dataset_roundtrip_preserves_command_intent_contract(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        load_distillation_dataset,
        save_distillation_dataset,
    )

    student_obs = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    teacher_obs = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    commands = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.0, 0.0, 0.20],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    command_intents = ("inactive", "active", "active", "inactive")
    dataset = build_distillation_dataset(
        student_obs,
        teacher_obs,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
        metadata={"source": "command-intent-fixture"},
        commands=commands,
        command_intents=command_intents,
        role_labels=("stand", "walk_flat", "walk_flat", "stand"),
    )

    batch = dataset.as_batch(start=1, batch_size=2)
    assert batch.commands is not None
    assert torch.equal(batch.commands, commands[1:3])
    assert batch.command_intents == ("active", "active")
    assert batch.role_labels == ("walk_flat", "walk_flat")

    checkpoint_path = tmp_path / "command_intent_distill_dataset.pt"
    save_distillation_dataset(checkpoint_path, dataset)
    restored = load_distillation_dataset(
        checkpoint_path,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
    )

    assert restored.commands is not None
    assert torch.equal(restored.commands, commands)
    assert restored.command_intents == command_intents
    assert restored.metadata["command_intents"] == list(command_intents)
    assert restored.metadata["command_intent_counts"] == {"active": 2, "inactive": 2}


def test_distillation_dataset_roundtrip_preserves_cached_teacher_actions(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        load_distillation_dataset,
        save_distillation_dataset,
    )

    student_obs = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    teacher_obs = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    teacher_actions = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    dataset = build_distillation_dataset(
        student_obs,
        teacher_obs,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
        expected_teacher_action_dim=3,
        metadata={"source": "cached-action-fixture"},
        role_labels=("stand", "walk", "height_low", "height_high"),
        teacher_actions=teacher_actions,
    )

    batch = dataset.as_batch(start=1, batch_size=2)
    assert batch.teacher_actions is not None
    assert batch.teacher_actions.shape == (2, 3)
    assert torch.equal(batch.teacher_actions[0], teacher_actions[1])
    assert batch.role_labels == ("walk", "height_low")

    checkpoint_path = tmp_path / "cached_action_distill_dataset.pt"
    save_distillation_dataset(checkpoint_path, dataset)
    restored = load_distillation_dataset(
        checkpoint_path,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
        expected_teacher_action_dim=3,
    )

    assert restored.teacher_action_dim == 3
    assert restored.teacher_actions is not None
    assert torch.equal(restored.teacher_actions, teacher_actions)


def test_distillation_dataset_roundtrip_preserves_transition_schema(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        load_distillation_dataset,
        save_distillation_dataset,
    )

    student_obs = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    teacher_obs = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    transition_ages = torch.tensor([-1, -1, 0, 1], dtype=torch.int64)
    command_before = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.4, 0.0, 0.0]],
        dtype=torch.float32,
    )
    command_after = torch.zeros(4, 3, dtype=torch.float32)
    dataset = build_distillation_dataset(
        student_obs,
        teacher_obs,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
        expected_teacher_action_dim=3,
        teacher_actions=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        role_labels=("stand", "walk_flat", "walk_flat", "stand"),
        scenario_labels=("static_stand", "walk_flat", "walk_to_stop", "walk_to_stop"),
        transition_ages=transition_ages,
        command_before=command_before,
        command_after=command_after,
    )

    batch = dataset.as_batch(start=2, batch_size=2)
    assert batch.scenario_labels == ("walk_to_stop", "walk_to_stop")
    assert batch.transition_ages is not None
    assert torch.equal(batch.transition_ages, transition_ages[2:])
    assert batch.command_before is not None
    assert torch.equal(batch.command_before, command_before[2:])
    assert batch.command_after is not None
    assert torch.equal(batch.command_after, command_after[2:])

    checkpoint_path = tmp_path / "transition_distill_dataset.pt"
    save_distillation_dataset(checkpoint_path, dataset)
    restored = load_distillation_dataset(
        checkpoint_path,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
        expected_teacher_action_dim=3,
    )

    assert restored.scenario_labels == dataset.scenario_labels
    assert restored.transition_ages is not None
    assert torch.equal(restored.transition_ages, transition_ages)
    assert restored.command_before is not None
    assert torch.equal(restored.command_before, command_before)
    assert restored.command_after is not None
    assert torch.equal(restored.command_after, command_after)
    assert restored.metadata["transition_schema"] == "DISTILL-TRAIN-v002"
    assert restored.metadata["scenario_counts"] == {
        "static_stand": 1,
        "walk_flat": 1,
        "walk_to_stop": 2,
    }


def test_distillation_dataset_rejects_malformed_transition_schema() -> None:
    from unilab.algos.torch.distill import build_distillation_dataset

    base_kwargs = {
        "expected_student_obs_dim": 5,
        "expected_teacher_obs_dim": 7,
    }
    with pytest.raises(ValueError, match="require scenario_labels"):
        build_distillation_dataset(
            torch.zeros(2, 5),
            torch.zeros(2, 7),
            transition_ages=torch.tensor([-1, 0], dtype=torch.int64),
            **base_kwargs,
        )
    with pytest.raises(ValueError, match="integer dtype"):
        build_distillation_dataset(
            torch.zeros(2, 5),
            torch.zeros(2, 7),
            scenario_labels=("walk_to_stop", "walk_to_stop"),
            transition_ages=torch.tensor([0.0, 1.0]),
            command_before=torch.ones(2, 3),
            command_after=torch.zeros(2, 3),
            **base_kwargs,
        )
    with pytest.raises(ValueError, match="transition_age=-1"):
        build_distillation_dataset(
            torch.zeros(2, 5),
            torch.zeros(2, 7),
            scenario_labels=("static_stand", "walk_flat"),
            transition_ages=torch.tensor([0, -1], dtype=torch.int64),
            **base_kwargs,
        )
    with pytest.raises(ValueError, match="provided together"):
        build_distillation_dataset(
            torch.zeros(2, 5),
            torch.zeros(2, 7),
            scenario_labels=("walk_to_stop", "walk_to_stop"),
            transition_ages=torch.tensor([0, 1], dtype=torch.int64),
            command_before=torch.ones(2, 3),
            **base_kwargs,
        )
    with pytest.raises(ValueError, match="command_after must be zero"):
        build_distillation_dataset(
            torch.zeros(2, 5),
            torch.zeros(2, 7),
            scenario_labels=("walk_to_stop", "walk_to_stop"),
            transition_ages=torch.tensor([0, 1], dtype=torch.int64),
            command_before=torch.ones(2, 3),
            command_after=torch.ones(2, 3),
            **base_kwargs,
        )


def test_distillation_dataset_rejects_bad_obs_contract() -> None:
    from unilab.algos.torch.distill import build_distillation_dataset

    with pytest.raises(ValueError, match="batch size"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(3, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
        )

    with pytest.raises(ValueError, match="student_obs dim"):
        build_distillation_dataset(
            torch.zeros(4, 6),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
        )

    teacher_obs = torch.zeros(4, 7)
    teacher_obs[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            teacher_obs,
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
        )


def test_distillation_dataset_rejects_bad_role_labels_contract() -> None:
    from unilab.algos.torch.distill import build_distillation_dataset

    with pytest.raises(ValueError, match="role_labels length"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            role_labels=("stand",),
        )

    with pytest.raises(ValueError, match="empty labels"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            role_labels=("stand", "walk", "", "height"),
        )


def test_distillation_dataset_rejects_bad_command_intent_contract() -> None:
    from unilab.algos.torch.distill import build_distillation_dataset

    with pytest.raises(ValueError, match="commands.*shape"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            commands=torch.zeros(4, 2),
        )

    with pytest.raises(ValueError, match="commands batch size"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            commands=torch.zeros(3, 3),
        )

    commands = torch.zeros(4, 3)
    commands[0, 0] = float("nan")
    with pytest.raises(ValueError, match="commands.*finite"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            commands=commands,
        )

    with pytest.raises(ValueError, match="command_intents length"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            command_intents=("active",),
        )

    with pytest.raises(ValueError, match="command_intents.*active/inactive"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            command_intents=("active", "inactive", "walk", "stand"),
        )


def test_distillation_dataset_rejects_bad_cached_teacher_actions_contract() -> None:
    from unilab.algos.torch.distill import build_distillation_dataset

    with pytest.raises(ValueError, match="teacher_actions dim"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            expected_teacher_action_dim=3,
            teacher_actions=torch.zeros(4, 4),
        )

    with pytest.raises(ValueError, match="teacher action dataset batch size"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            expected_teacher_action_dim=3,
            teacher_actions=torch.zeros(3, 3),
        )

    teacher_actions = torch.zeros(4, 3)
    teacher_actions[0, 0] = float("nan")
    with pytest.raises(ValueError, match="teacher_actions.*finite"):
        build_distillation_dataset(
            torch.zeros(4, 5),
            torch.zeros(4, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            expected_teacher_action_dim=3,
            teacher_actions=teacher_actions,
        )


def test_multitask_distillation_dataset_adapter_merges_roles_and_cached_targets(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("UNILAB_DISTILL_RUNTIME_DEBUG", "1")
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        build_multitask_distillation_dataset,
        load_distillation_dataset,
        save_distillation_dataset,
    )

    stand_path = tmp_path / "stand.pt"
    walk_path = tmp_path / "walk.pt"
    stand_dataset = build_distillation_dataset(
        torch.full((2, 5), 1.0),
        torch.full((2, 5), 2.0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=5,
        expected_teacher_action_dim=3,
        teacher_actions=torch.full((2, 3), 0.25),
        commands=torch.zeros(2, 3),
        command_intents=("inactive", "inactive"),
        metadata={"task_name": "G1StandStill"},
    )
    walk_dataset = build_distillation_dataset(
        torch.full((3, 5), 3.0),
        torch.full((3, 5), 4.0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=5,
        expected_teacher_action_dim=3,
        teacher_actions=torch.full((3, 3), -0.5),
        commands=torch.tensor(
            [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]],
            dtype=torch.float32,
        ),
        command_intents=("active", "active", "active"),
        metadata={"task_name": "G1WalkHeight"},
    )
    save_distillation_dataset(stand_path, stand_dataset)
    save_distillation_dataset(walk_path, walk_dataset)

    merged = build_multitask_distillation_dataset(
        [
            {"path": stand_path, "role": "stand"},
            {"path": walk_path, "role": "walk_height"},
        ],
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=5,
        expected_teacher_action_dim=3,
    )

    assert merged.num_samples == 5
    assert merged.role_labels == (
        "stand",
        "stand",
        "walk_height",
        "walk_height",
        "walk_height",
    )
    assert merged.teacher_action_dim == 3
    assert merged.teacher_actions is not None
    assert torch.allclose(merged.teacher_actions[:2], torch.full((2, 3), 0.25))
    assert torch.allclose(merged.teacher_actions[2:], torch.full((3, 3), -0.5))
    assert merged.metadata["source"] == "multitask_adapter"
    assert merged.metadata["source_count"] == 2
    assert merged.metadata["source_roles"] == ["stand", "walk_height"]
    assert merged.metadata["source_sample_counts"] == [2, 3]
    assert merged.commands is not None
    assert merged.command_intents == (
        "inactive",
        "inactive",
        "active",
        "active",
        "active",
    )
    assert merged.metadata["command_intent_counts"] == {"active": 3, "inactive": 2}
    assert merged.as_batch(start=1, batch_size=3).role_labels == (
        "stand",
        "walk_height",
        "walk_height",
    )

    roundtrip_path = tmp_path / "merged.pt"
    save_distillation_dataset(roundtrip_path, merged)
    reloaded = load_distillation_dataset(
        roundtrip_path,
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=5,
        expected_teacher_action_dim=3,
    )
    assert reloaded.role_labels == merged.role_labels
    assert reloaded.teacher_actions is not None
    assert torch.allclose(reloaded.teacher_actions, merged.teacher_actions)
    assert reloaded.commands is not None
    assert torch.equal(reloaded.commands, merged.commands)
    assert reloaded.command_intents == merged.command_intents

    prefix = "[distill-data-runtime] "
    snapshots = [
        ast.literal_eval(line.removeprefix(prefix))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(prefix)
    ]
    multitask_stages = [
        snapshot["stage"] for snapshot in snapshots if snapshot["stage"].startswith("multitask/")
    ]
    assert multitask_stages == [
        "multitask/entry",
        "multitask/before_source_load",
        "multitask/after_source_annotation",
        "multitask/before_source_load",
        "multitask/after_source_annotation",
        "multitask/after_concat",
        "multitask/before_final_validation",
        "multitask/after_final_validation",
    ]


def test_distillation_data_runtime_trace_wraps_torch_save_and_load(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("UNILAB_DISTILL_RUNTIME_DEBUG", "1")
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        load_distillation_dataset,
        save_distillation_dataset,
    )

    dataset = build_distillation_dataset(
        torch.zeros(2, 3),
        torch.zeros(2, 4),
        expected_student_obs_dim=3,
        expected_teacher_obs_dim=4,
    )
    path = tmp_path / "runtime-trace.pt"
    save_distillation_dataset(path, dataset)
    load_distillation_dataset(path)

    prefix = "[distill-data-runtime] "
    snapshots = [
        ast.literal_eval(line.removeprefix(prefix))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(prefix)
    ]
    serialization = [
        snapshot for snapshot in snapshots if snapshot["stage"].startswith("serialization/")
    ]
    assert [snapshot["stage"] for snapshot in serialization] == [
        "serialization/before_torch_save",
        "serialization/after_torch_save",
        "serialization/before_torch_load",
        "serialization/after_torch_load",
    ]
    assert all(snapshot["torch_is_storage_callable"] is True for snapshot in serialization)
    assert all(snapshot["builtins_int_callable"] is True for snapshot in serialization)
    assert serialization[0]["path"] == str(path)
    assert serialization[-1]["payload_keys"] == sorted(
        [
            "command_after",
            "command_before",
            "command_intents",
            "commands",
            "metadata",
            "num_samples",
            "role_labels",
            "scenario_labels",
            "student_obs",
            "student_obs_dim",
            "teacher_action_dim",
            "teacher_actions",
            "teacher_obs",
            "teacher_obs_dim",
            "transition_ages",
        ]
    )


def test_command_intent_corruption_requests_native_abort_with_snapshot(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("UNILAB_DISTILL_RUNTIME_DEBUG", "1")
    import unilab.algos.torch.distill.data as data_module

    class ImpossibleIntent:
        def __str__(self) -> str:
            return "impossible"

        def __repr__(self) -> str:
            return "<impossible-intent>"

    class NativeAbortRequestedError(RuntimeError):
        pass

    def request_abort() -> None:
        raise NativeAbortRequestedError

    monkeypatch.setenv("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "1")
    monkeypatch.setattr(data_module.os, "abort", request_abort)

    with pytest.raises(NativeAbortRequestedError):
        data_module._validate_command_intents(  # noqa: SLF001 - diagnostic contract
            ["active", ImpossibleIntent()],
            num_samples=2,
        )

    prefix = "[distill-data-runtime] "
    snapshots = [
        ast.literal_eval(line.removeprefix(prefix))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(prefix)
    ]
    failure = snapshots[-1]
    assert failure["stage"] == "command_intent_validation/corruption_detected"
    assert failure["invalid_count"] == 1
    assert failure["invalid_head"] == [
        {
            "index": 1,
            "raw_type": "ImpossibleIntent",
            "raw_repr": "<impossible-intent>",
            "normalized": "impossible",
        }
    ]
    assert failure["native_abort_requested"] is True


def test_serialization_callable_corruption_requests_native_abort(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("UNILAB_DISTILL_RUNTIME_DEBUG", "1")
    import unilab.algos.torch.distill.data as data_module
    from unilab.algos.torch.distill import build_distillation_dataset

    class NativeAbortRequestedError(RuntimeError):
        pass

    dataset = build_distillation_dataset(
        torch.zeros(2, 3),
        torch.zeros(2, 4),
        expected_student_obs_dim=3,
        expected_teacher_obs_dim=4,
    )
    monkeypatch.setenv("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "1")
    monkeypatch.setattr(
        data_module.torch,
        "save",
        lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("'cell' object is not callable")),
    )
    monkeypatch.setattr(
        data_module,
        "_abort_for_native_capture",
        lambda: (_ for _ in ()).throw(NativeAbortRequestedError),
    )

    with pytest.raises(NativeAbortRequestedError):
        data_module.save_distillation_dataset(tmp_path / "corrupt.pt", dataset)

    prefix = "[distill-data-runtime] "
    snapshots = [
        ast.literal_eval(line.removeprefix(prefix))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(prefix)
    ]
    failure = snapshots[-1]
    assert failure["stage"] == "serialization/torch_save_failure"
    assert failure["error_type"] == "TypeError"
    assert failure["native_abort_requested"] is True
    assert failure["builtins_isinstance_callable"] is True
    assert failure["builtins_isinstance_is_original"] is True


def test_serialization_io_failure_does_not_request_native_abort(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("UNILAB_DISTILL_RUNTIME_DEBUG", "1")
    import unilab.algos.torch.distill.data as data_module
    from unilab.algos.torch.distill import build_distillation_dataset

    dataset = build_distillation_dataset(
        torch.zeros(2, 3),
        torch.zeros(2, 4),
        expected_student_obs_dim=3,
        expected_teacher_obs_dim=4,
    )
    monkeypatch.setenv("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "1")
    monkeypatch.setattr(
        data_module.torch,
        "save",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        data_module,
        "_abort_for_native_capture",
        lambda: pytest.fail("ordinary IO failure must not request native abort"),
    )

    with pytest.raises(OSError, match="disk full"):
        data_module.save_distillation_dataset(tmp_path / "io-error.pt", dataset)

    prefix = "[distill-data-runtime] "
    snapshots = [
        ast.literal_eval(line.removeprefix(prefix))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(prefix)
    ]
    assert snapshots[-1]["stage"] == "serialization/torch_save_failure"
    assert snapshots[-1]["native_abort_requested"] is False


def test_multitask_command_intent_failure_emits_source_provenance_snapshot(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import unilab.algos.torch.distill.data as data_module
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        build_multitask_distillation_dataset,
        save_distillation_dataset,
    )

    source_paths = []
    for role, intent in (("stand", "inactive"), ("walk", "active")):
        path = tmp_path / f"{role}.pt"
        dataset = build_distillation_dataset(
            torch.zeros(2, 5),
            torch.zeros(2, 7),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            expected_teacher_action_dim=3,
            teacher_actions=torch.zeros(2, 3),
            commands=torch.zeros(2, 3),
            command_intents=(intent, intent),
        )
        save_distillation_dataset(path, dataset)
        source_paths.append((path, role))

    real_builder = data_module.build_distillation_dataset
    call_count = 0

    def fail_only_at_final_validation(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise ValueError("command_intents must contain only active/inactive labels")
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(
        data_module,
        "build_distillation_dataset",
        fail_only_at_final_validation,
    )
    with pytest.raises(ValueError, match="command_intents.*active/inactive"):
        build_multitask_distillation_dataset(
            [{"path": path, "role": role} for path, role in source_paths],
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=7,
            expected_teacher_action_dim=3,
        )

    output_lines = capsys.readouterr().out.splitlines()
    snapshots = [
        json.loads(line.removeprefix("[distill-command-intent-sentinel] "))
        for line in output_lines
        if line.startswith("[distill-command-intent-sentinel] ")
    ]
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["stage"] == "multitask/final_validation_failure"
    assert snapshot["source_count"] == 2
    assert snapshot["sources"] == [
        {
            "command_intent_counts": {"inactive": 2},
            "invalid_head": [],
            "num_samples": 2,
            "path": str(source_paths[0][0]),
            "role": "stand",
            "scenario": None,
        },
        {
            "command_intent_counts": {"active": 2},
            "invalid_head": [],
            "num_samples": 2,
            "path": str(source_paths[1][0]),
            "role": "walk",
            "scenario": None,
        },
    ]
    assert snapshot["before_final_validation"] == {
        "command_intent_counts": {"active": 2, "inactive": 2},
        "invalid_head": [],
        "length": 4,
        "type": "tuple",
    }
    assert snapshot["after_final_validation_failure"] == snapshot["before_final_validation"]


def test_multitask_distillation_dataset_adapter_fails_closed(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        build_multitask_distillation_dataset,
        save_distillation_dataset,
    )

    no_action_path = tmp_path / "no_action.pt"
    bad_dim_path = tmp_path / "bad_dim.pt"
    save_distillation_dataset(
        no_action_path,
        build_distillation_dataset(
            torch.zeros(2, 5),
            torch.zeros(2, 5),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
        ),
    )
    save_distillation_dataset(
        bad_dim_path,
        build_distillation_dataset(
            torch.zeros(2, 6),
            torch.zeros(2, 5),
            expected_student_obs_dim=6,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
            teacher_actions=torch.zeros(2, 3),
        ),
    )
    matching_path = tmp_path / "matching.pt"
    save_distillation_dataset(
        matching_path,
        build_distillation_dataset(
            torch.zeros(2, 5),
            torch.zeros(2, 5),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
            teacher_actions=torch.zeros(2, 3),
        ),
    )
    command_schema_path = tmp_path / "command_schema.pt"
    save_distillation_dataset(
        command_schema_path,
        build_distillation_dataset(
            torch.zeros(2, 5),
            torch.zeros(2, 5),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
            teacher_actions=torch.zeros(2, 3),
            commands=torch.zeros(2, 3),
            command_intents=("inactive", "inactive"),
        ),
    )

    with pytest.raises(ValueError, match="at least one source"):
        build_multitask_distillation_dataset([])
    with pytest.raises(ValueError, match="role"):
        build_multitask_distillation_dataset([{"path": no_action_path}])
    with pytest.raises(ValueError, match="cached teacher_actions"):
        build_multitask_distillation_dataset(
            [{"path": no_action_path, "role": "stand"}],
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
        )
    with pytest.raises(ValueError, match="student_obs dim mismatch"):
        build_multitask_distillation_dataset(
            [{"path": bad_dim_path, "role": "walk_height"}],
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
        )
    with pytest.raises(ValueError, match="multitask source .* student_obs dim mismatch"):
        build_multitask_distillation_dataset(
            [
                {"path": matching_path, "role": "stand"},
                {"path": bad_dim_path, "role": "walk_height"},
            ],
        )
    with pytest.raises(ValueError, match="all include commands or none"):
        build_multitask_distillation_dataset(
            [
                {"path": matching_path, "role": "stand"},
                {"path": command_schema_path, "role": "walk_height"},
            ],
        )


def test_multitask_distillation_dataset_merges_transition_fields(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("UNILAB_DISTILL_RUNTIME_DEBUG", "1")
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        build_multitask_distillation_dataset,
        save_distillation_dataset,
    )

    stand_path = tmp_path / "stand_transition.pt"
    transition_path = tmp_path / "transition.pt"
    common = {
        "expected_student_obs_dim": 5,
        "expected_teacher_obs_dim": 5,
        "expected_teacher_action_dim": 3,
        "teacher_actions": torch.zeros(2, 3),
        "transition_ages": torch.tensor([-1, -1], dtype=torch.int64),
        "command_before": torch.zeros(2, 3),
        "command_after": torch.zeros(2, 3),
    }
    save_distillation_dataset(
        stand_path,
        build_distillation_dataset(
            torch.zeros(2, 5),
            torch.zeros(2, 5),
            role_labels=("stand", "stand"),
            scenario_labels=("static_stand", "static_stand"),
            **common,
        ),
    )
    save_distillation_dataset(
        transition_path,
        build_distillation_dataset(
            torch.ones(2, 5),
            torch.ones(2, 5),
            role_labels=("walk_flat", "stand"),
            scenario_labels=("walk_to_stop", "walk_to_stop"),
            transition_ages=torch.tensor([0, 1], dtype=torch.int64),
            command_before=torch.full((2, 3), 0.4),
            command_after=torch.zeros(2, 3),
            teacher_actions=torch.ones(2, 3),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
        ),
    )

    merged = build_multitask_distillation_dataset(
        [
            {"path": stand_path, "role": "stand"},
            {"path": transition_path, "role": "walk_to_stop"},
        ],
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=5,
        expected_teacher_action_dim=3,
    )

    assert merged.scenario_labels == (
        "static_stand",
        "static_stand",
        "walk_to_stop",
        "walk_to_stop",
    )
    assert merged.transition_ages is not None
    assert torch.equal(merged.transition_ages, torch.tensor([-1, -1, 0, 1]))
    assert merged.command_before is not None
    assert torch.equal(merged.command_before[:2], torch.zeros(2, 3))
    assert merged.command_after is not None
    assert torch.equal(merged.command_after[2:], torch.zeros(2, 3))

    prefix = "[distill-data-runtime] "
    snapshots = [
        ast.literal_eval(line.removeprefix(prefix))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(prefix)
    ]
    scenario_snapshots = [
        snapshot for snapshot in snapshots if snapshot["stage"].startswith("multitask/scenario_")
    ]
    assert [snapshot["stage"] for snapshot in scenario_snapshots] == [
        "multitask/scenario_source_ready",
        "multitask/scenario_source_ready",
        "multitask/scenario_concat_chunk",
        "multitask/scenario_concat_chunk",
        "multitask/scenario_concat_complete",
    ]
    assert scenario_snapshots[0]["path"] == str(stand_path)
    assert scenario_snapshots[0]["scenario_labels"]["label_counts"] == {"static_stand": 2}
    assert scenario_snapshots[2]["global_start"] == 0
    assert scenario_snapshots[2]["global_stop"] == 2
    assert scenario_snapshots[2]["observation_timing"] == "post_flatten_slice_check"
    assert scenario_snapshots[2]["source_matches_aggregate_slice"] is True
    assert scenario_snapshots[3]["path"] == str(transition_path)
    assert scenario_snapshots[3]["global_start"] == 2
    assert scenario_snapshots[3]["global_stop"] == 4
    assert scenario_snapshots[3]["source_matches_aggregate_slice"] is True
    assert scenario_snapshots[-1]["scenario_labels"]["label_counts"] == {
        "static_stand": 2,
        "walk_to_stop": 2,
    }
    assert scenario_snapshots[-1]["scenario_labels"]["invalid_head"] == []
    assert all(snapshot["builtins_str_is_original"] is True for snapshot in snapshots)
    assert all(snapshot["builtins_type_is_original"] is True for snapshot in snapshots)
    assert all(snapshot["builtins_tuple_is_original"] is True for snapshot in snapshots)
    assert all(snapshot["builtins_list_is_original"] is True for snapshot in snapshots)


def test_multitask_scenario_failure_emits_raw_source_provenance_snapshot(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from dataclasses import replace

    import unilab.algos.torch.distill.data as data_module
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        build_multitask_distillation_dataset,
        save_distillation_dataset,
    )

    source_path = tmp_path / "transition.pt"
    source = build_distillation_dataset(
        torch.zeros(2, 5),
        torch.zeros(2, 5),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=5,
        expected_teacher_action_dim=3,
        teacher_actions=torch.zeros(2, 3),
        scenario_labels=("walk_to_stop", "walk_to_stop"),
        transition_ages=torch.tensor([0, 1], dtype=torch.int64),
        command_before=torch.ones(2, 3),
        command_after=torch.zeros(2, 3),
    )
    save_distillation_dataset(source_path, source)

    real_load = data_module.load_distillation_dataset

    def load_with_runtime_corruption(*args, **kwargs):
        loaded = real_load(*args, **kwargs)
        return replace(
            loaded,
            scenario_labels=("walk_to_stop", types.FrameType),
        )

    monkeypatch.setattr(
        data_module,
        "load_distillation_dataset",
        load_with_runtime_corruption,
    )
    with pytest.raises(ValueError, match=r"scenario_labels.*<class 'frame'>"):
        build_multitask_distillation_dataset(
            [{"path": source_path, "role": "walk_flat"}],
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
        )

    output_lines = capsys.readouterr().out.splitlines()
    snapshots = [
        json.loads(line.removeprefix("[distill-scenario-label-sentinel] "))
        for line in output_lines
        if line.startswith("[distill-scenario-label-sentinel] ")
    ]
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["stage"] == "multitask/final_validation_failure"
    assert snapshot["source_count"] == 1
    assert snapshot["aggregate"]["invalid_head"] == [
        {
            "global_index": 1,
            "normalized": "<class 'frame'>",
            "path": str(source_path),
            "raw_repr": "<class 'frame'>",
            "raw_type": "type",
            "role": "walk_flat",
            "scenario": None,
            "source_index": 0,
            "source_row_index": 1,
        }
    ]
    assert snapshot["sources"][0]["scenario_labels"]["invalid_head"] == [
        {
            "index": 1,
            "normalized": "<class 'frame'>",
            "raw_repr": "<class 'frame'>",
            "raw_type": "type",
        }
    ]


def test_multitask_source_annotation_failure_reports_source_context(
    tmp_path,
    capsys,
) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        build_multitask_distillation_dataset,
        save_distillation_dataset,
    )

    source_path = tmp_path / "walk_flat_bad_intent.pt"
    save_distillation_dataset(
        source_path,
        build_distillation_dataset(
            torch.zeros(3, 5),
            torch.zeros(3, 5),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
            metadata={
                "command_sample_filter": "active",
                "command_seen_samples": 5,
                "command_selected_samples": 3,
                "command_intent_counts": {"inactive": 1, "active": 2},
            },
            teacher_actions=torch.zeros(3, 3),
            commands=torch.zeros(3, 3),
            command_intents=("active", "inactive", "active"),
        ),
    )

    with pytest.raises(
        ValueError,
        match="multitask source scenario annotation failed",
    ) as exc_info:
        build_multitask_distillation_dataset(
            [
                {
                    "source_index": 7,
                    "path": source_path,
                    "role": "walk_flat",
                    "scenario": "walk_flat",
                }
            ],
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
        )

    assert str(source_path) in str(exc_info.value)
    assert '"source_index": 7' in str(exc_info.value)
    assert '"requested_scenario": "walk_flat"' in str(exc_info.value)
    assert '"expected_intent": "active"' in str(exc_info.value)
    assert '"index": 1' in str(exc_info.value)

    output_lines = capsys.readouterr().out.splitlines()
    snapshots = [
        json.loads(line.removeprefix("[distill-source-annotation-sentinel] "))
        for line in output_lines
        if line.startswith("[distill-source-annotation-sentinel] ")
    ]
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["stage"] == "multitask/source_annotation_failure"
    assert snapshot["source_index"] == 7
    assert snapshot["path"] == str(source_path)
    assert snapshot["role"] == "walk_flat"
    assert snapshot["requested_scenario"] == "walk_flat"
    assert snapshot["command_intents"]["expected_intent"] == "active"
    assert snapshot["command_intents"]["expected_mismatch_head"] == [
        {
            "index": 1,
            "normalized": "inactive",
            "repr": "'inactive'",
            "type": "str",
        }
    ]
    assert snapshot["metadata"] == {
        "command_intent_counts": {"active": 2, "inactive": 1},
        "command_sample_filter": "active",
        "command_seen_samples": 5,
        "command_selected_samples": 3,
    }


def test_multitask_uses_dataset_workflow_scenario_metadata_as_owner_contract(
    tmp_path,
) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        build_multitask_distillation_dataset,
        save_distillation_dataset,
    )

    source_path = tmp_path / "walk_flat_metadata_owned.pt"
    save_distillation_dataset(
        source_path,
        build_distillation_dataset(
            torch.zeros(2, 5),
            torch.zeros(2, 5),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
            metadata={"workflow_scenario": "walk_flat"},
            teacher_actions=torch.zeros(2, 3),
            commands=torch.full((2, 3), 0.4),
            command_intents=("active", "active"),
        ),
    )

    merged = build_multitask_distillation_dataset(
        [{"path": source_path, "role": "walk_flat"}],
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=5,
        expected_teacher_action_dim=3,
    )

    assert merged.scenario_labels == ("walk_flat", "walk_flat")
    assert merged.metadata["source_scenarios"] == ["walk_flat"]


def test_multitask_rejects_source_scenario_metadata_drift(
    tmp_path,
) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        build_multitask_distillation_dataset,
        save_distillation_dataset,
    )

    source_path = tmp_path / "static_stand_metadata_owned.pt"
    save_distillation_dataset(
        source_path,
        build_distillation_dataset(
            torch.zeros(2, 5),
            torch.zeros(2, 5),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
            metadata={"workflow_scenario": "static_stand"},
            teacher_actions=torch.zeros(2, 3),
            commands=torch.zeros(2, 3),
            command_intents=("inactive", "inactive"),
        ),
    )

    with pytest.raises(
        ValueError,
        match="multitask source scenario contract mismatch",
    ) as exc_info:
        build_multitask_distillation_dataset(
            [
                {
                    "path": source_path,
                    "role": "walk_flat",
                    "scenario": "walk_flat",
                }
            ],
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=5,
            expected_teacher_action_dim=3,
        )

    assert '"metadata_workflow_scenario": "static_stand"' in str(exc_info.value)
    assert '"requested_scenario": "walk_flat"' in str(exc_info.value)


def test_multitask_workflow_scenario_annotation_preserves_row_roles(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        build_multitask_distillation_dataset,
        save_distillation_dataset,
    )

    stand_path = tmp_path / "stand_legacy.pt"
    walk_path = tmp_path / "walk_legacy.pt"
    save_distillation_dataset(
        stand_path,
        build_distillation_dataset(
            torch.zeros(2, 5),
            torch.empty(2, 0),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=0,
            expected_teacher_action_dim=3,
            teacher_actions=torch.zeros(2, 3),
            role_labels=("stand", "stand"),
            command_intents=("inactive", "inactive"),
        ),
    )
    save_distillation_dataset(
        walk_path,
        build_distillation_dataset(
            torch.ones(2, 5),
            torch.empty(2, 0),
            expected_student_obs_dim=5,
            expected_teacher_obs_dim=0,
            expected_teacher_action_dim=3,
            teacher_actions=torch.ones(2, 3),
            role_labels=("walk_flat", "walk_flat"),
            commands=torch.full((2, 3), 0.4),
            command_intents=("active", "active"),
        ),
    )

    merged = build_multitask_distillation_dataset(
        [
            {
                "path": stand_path,
                "role": "stand",
                "scenario": "static_stand",
                "preserve_row_role_labels": True,
            },
            {
                "path": walk_path,
                "role": "walk_flat",
                "scenario": "walk_flat",
                "preserve_row_role_labels": True,
            },
        ],
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
    )

    assert merged.role_labels == ("stand", "stand", "walk_flat", "walk_flat")
    assert merged.scenario_labels == (
        "static_stand",
        "static_stand",
        "walk_flat",
        "walk_flat",
    )
    assert merged.transition_ages is not None
    assert torch.equal(merged.transition_ages, torch.full((4,), -1, dtype=torch.int64))
    assert merged.metadata["source_scenarios"] == ["static_stand", "walk_flat"]


def test_command_active_mask_marks_any_velocity_command_active() -> None:
    from unilab.algos.torch.distill import command_active_mask

    commands = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.06, 0.0, 0.0],
            [0.0, -0.06, 0.0],
            [0.03, 0.04, 0.0],
            [0.04, 0.04, 0.0],
            [0.0, 0.0, 0.06],
            [0.0, 0.0, -0.06],
        ],
        dtype=np.float32,
    )

    mask = command_active_mask(commands, xy_threshold=0.05, yaw_threshold=0.05)

    np.testing.assert_array_equal(
        mask,
        np.asarray([False, True, True, False, True, True, True], dtype=np.bool_),
    )


@pytest.mark.parametrize(
    "commands",
    [
        np.zeros((3,), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 4), dtype=np.float32),
        np.asarray([[0.0, np.nan, 0.0]], dtype=np.float32),
    ],
)
def test_command_active_mask_fails_closed_for_bad_commands(commands: np.ndarray) -> None:
    from unilab.algos.torch.distill import command_active_mask

    with pytest.raises(ValueError, match="commands"):
        command_active_mask(commands, xy_threshold=0.05, yaw_threshold=0.05)


@pytest.mark.parametrize(
    ("xy_threshold", "yaw_threshold"),
    [
        (-0.01, 0.05),
        (0.05, -0.01),
        (np.inf, 0.05),
        (0.05, np.nan),
    ],
)
def test_command_active_mask_fails_closed_for_bad_thresholds(
    xy_threshold: float,
    yaw_threshold: float,
) -> None:
    from unilab.algos.torch.distill import command_active_mask

    with pytest.raises(ValueError, match="threshold"):
        command_active_mask(
            np.zeros((1, 3), dtype=np.float32),
            xy_threshold=xy_threshold,
            yaw_threshold=yaw_threshold,
        )


class _FakeDistillEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.action_space = type("ActionSpace", (), {"shape": (3,)})()
        self.reset_calls = 0
        self.step_calls = 0
        self.state = None
        self.last_actions = None

    def init_state(self) -> None:
        self.state = object()

    def reset(self, env_indices):
        self.reset_calls += 1
        return self._obs(0), {"reset_indices": np.asarray(env_indices)}

    def step(self, actions):
        self.step_calls += 1
        assert actions.shape == (2, 3)
        self.last_actions = np.asarray(actions, dtype=np.float32)
        return type("State", (), {"obs": self._obs(self.step_calls), "info": {}})()

    def _obs(self, offset: int) -> dict[str, np.ndarray]:
        base = np.arange(16, dtype=np.float32).reshape(2, 8) + float(offset)
        return {"obs": base, "critic": base + 100.0}


class _CommandInfoDistillEnv(_FakeDistillEnv):
    def __init__(self, command_batches: list[np.ndarray]) -> None:
        super().__init__()
        self.command_batches = command_batches

    def reset(self, env_indices):
        obs, _info = super().reset(env_indices)
        return obs, {"commands": self.command_batches[0]}

    def step(self, actions):
        state = super().step(actions)
        batch_index = min(self.step_calls, len(self.command_batches) - 1)
        return type(
            "State",
            (),
            {"obs": state.obs, "info": {"commands": self.command_batches[batch_index]}},
        )()


class _TransitionDistillEnv:
    def __init__(self, *, done_at_step: int | None = None) -> None:
        self.num_envs = 2
        self.action_space = type("ActionSpace", (), {"shape": (3,)})()
        self.done_at_step = done_at_step
        self.step_calls = 0
        self.reset_calls = 0
        self.commands = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.command_history: list[np.ndarray] = []
        self.action_history: list[np.ndarray] = []
        self.state = None

    def _obs(self, offset: int) -> dict[str, np.ndarray]:
        base = np.arange(16, dtype=np.float32).reshape(self.num_envs, 8)
        return {"obs": base + float(offset)}

    def _state(self, *, terminated: np.ndarray | None = None):
        return type(
            "State",
            (),
            {
                "obs": self._obs(self.step_calls),
                "info": {"commands": self.commands},
                "terminated": (
                    np.zeros((self.num_envs,), dtype=np.bool_) if terminated is None else terminated
                ),
                "truncated": np.zeros((self.num_envs,), dtype=np.bool_),
                "final_observation": None,
            },
        )()

    def init_state(self) -> None:
        self.state = self._state()

    def reset(self, env_indices):
        indices = np.asarray(env_indices, dtype=np.int32).reshape(-1)
        self.reset_calls += 1
        self.commands[indices] = 0.0
        self.state = self._state()
        reset_obs = {key: value[indices] for key, value in self._obs(0).items()}
        return reset_obs, {"commands": self.commands[indices].copy()}

    def refresh_state(self):
        self.command_history.append(self.commands.copy())
        self.state = self._state()
        return self.state

    def step(self, actions):
        assert actions.shape == (self.num_envs, 3)
        self.action_history.append(np.asarray(actions, dtype=np.float32).copy())
        self.step_calls += 1
        terminated = np.zeros((self.num_envs,), dtype=np.bool_)
        if self.done_at_step == self.step_calls:
            terminated[0] = True
        self.state = self._state(terminated=terminated)
        return self.state


def test_collect_distillation_dataset_from_env_projects_student_obs() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    dataset = collect_distillation_dataset_from_env(
        _FakeDistillEnv(),
        num_samples=3,
        expected_student_obs_dim=7,
        expected_teacher_obs_dim=8,
        teacher_obs_key="obs",
        student_projection="drop_index",
        student_drop_index=3,
        action_mode="zero",
    )

    assert dataset.num_samples == 3
    assert dataset.student_obs_dim == 7
    assert dataset.teacher_obs_dim == 8
    assert dataset.metadata["source"] == "live_env_rollout"
    assert dataset.metadata["student_projection"] == "drop_index"
    assert dataset.metadata["teacher_projection"] == "identity"
    assert dataset.metadata["student_drop_index"] == 3
    assert dataset.metadata["teacher_obs_key"] == "obs"
    assert dataset.metadata["action_mode"] == "zero"
    assert dataset.metadata["synthetic_teacher_tail"] is False
    assert "command_sample_filter" not in dataset.metadata
    assert torch.equal(dataset.teacher_obs[0], torch.arange(8, dtype=torch.float32))
    assert torch.equal(
        dataset.student_obs[0],
        torch.tensor([0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0]),
    )


def test_collect_transition_distillation_dataset_switches_teacher_and_command() -> None:
    from unilab.algos.torch.distill import (
        collect_transition_distillation_dataset_from_env,
    )

    class ConstantPolicy(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = float(value)

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return torch.full((obs.shape[0], 3), self.value, dtype=obs.dtype, device=obs.device)

    env = _TransitionDistillEnv()
    dataset = collect_transition_distillation_dataset_from_env(
        env,
        num_samples=8,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        walking_teacher_policy=ConstantPolicy(0.1),
        standing_teacher_policy=ConstantPolicy(0.2),
        rollout_policy=ConstantPolicy(-0.3),
        pre_switch_steps=2,
        walk_command=np.asarray([0.4, 0.0, 0.0], dtype=np.float32),
    )

    assert dataset.scenario_labels == ("walk_to_stop",) * 8
    assert dataset.role_labels == ("walk_flat",) * 4 + ("stand",) * 4
    assert dataset.command_intents == ("active",) * 4 + ("inactive",) * 4
    assert dataset.transition_ages is not None
    assert torch.equal(dataset.transition_ages, torch.tensor([-1, -1, -1, -1, 0, 0, 1, 1]))
    assert dataset.teacher_actions is not None
    assert torch.allclose(dataset.teacher_actions[:4], torch.full((4, 3), 0.1))
    assert torch.allclose(dataset.teacher_actions[4:], torch.full((4, 3), 0.2))
    assert dataset.command_before is not None
    assert torch.allclose(dataset.command_before, torch.tensor([[0.4, 0.0, 0.0]] * 8))
    assert dataset.command_after is not None
    assert torch.allclose(dataset.command_after[:4], torch.tensor([[0.4, 0.0, 0.0]] * 4))
    assert torch.equal(dataset.command_after[4:], torch.zeros(4, 3))
    assert len(env.command_history) == 2
    assert np.array_equal(
        env.command_history[0],
        np.full((2, 3), [0.4, 0.0, 0.0], dtype=np.float32),
    )
    assert np.array_equal(env.command_history[1], np.zeros((2, 3), dtype=np.float32))
    assert dataset.metadata["switch_count"] == 2
    assert dataset.metadata["post_switch_rows"] == 4


def test_collect_transition_distillation_dataset_switches_rollout_expert() -> None:
    from unilab.algos.torch.distill import (
        collect_transition_distillation_dataset_from_env,
    )

    class ConstantPolicy(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = float(value)

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return torch.full((obs.shape[0], 3), self.value, dtype=obs.dtype, device=obs.device)

    env = _TransitionDistillEnv()
    dataset = collect_transition_distillation_dataset_from_env(
        env,
        num_samples=8,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        walking_teacher_policy=ConstantPolicy(0.1),
        standing_teacher_policy=ConstantPolicy(0.2),
        rollout_policies_by_intent={
            "active": ConstantPolicy(0.3),
            "inactive": ConstantPolicy(0.4),
        },
        pre_switch_steps=2,
    )

    assert dataset.metadata["rollout_policy"] == "command_intent_experts"
    assert len(env.action_history) == 3
    assert np.allclose(env.action_history[0], 0.3)
    assert np.allclose(env.action_history[1], 0.3)
    assert np.allclose(env.action_history[2], 0.4)


def test_collect_transition_distillation_dataset_resets_done_rows() -> None:
    from unilab.algos.torch.distill import (
        collect_transition_distillation_dataset_from_env,
    )

    class ConstantPolicy(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return torch.zeros((obs.shape[0], 3), dtype=obs.dtype, device=obs.device)

    env = _TransitionDistillEnv(done_at_step=3)
    dataset = collect_transition_distillation_dataset_from_env(
        env,
        num_samples=10,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        walking_teacher_policy=ConstantPolicy(),
        standing_teacher_policy=ConstantPolicy(),
        rollout_policy=ConstantPolicy(),
        pre_switch_steps=2,
    )

    assert dataset.num_samples == 10
    assert dataset.metadata["done_seen_samples"] == 1
    assert env.reset_calls >= 2
    assert dataset.transition_ages is not None
    assert torch.any(dataset.transition_ages == -1)
    assert torch.any(dataset.transition_ages >= 0)


def test_collect_transition_distillation_dataset_enforces_post_switch_horizon() -> None:
    from unilab.algos.torch.distill import (
        collect_transition_distillation_dataset_from_env,
    )

    class ConstantPolicy(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return torch.zeros((obs.shape[0], 3), dtype=obs.dtype, device=obs.device)

    with pytest.raises(ValueError, match="minimum=10"):
        collect_transition_distillation_dataset_from_env(
            _TransitionDistillEnv(),
            num_samples=8,
            expected_student_obs_dim=8,
            expected_teacher_obs_dim=8,
            walking_teacher_policy=ConstantPolicy(),
            standing_teacher_policy=ConstantPolicy(),
            rollout_policy=ConstantPolicy(),
            pre_switch_steps=2,
            min_post_switch_steps=3,
        )

    dataset = collect_transition_distillation_dataset_from_env(
        _TransitionDistillEnv(),
        num_samples=10,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        walking_teacher_policy=ConstantPolicy(),
        standing_teacher_policy=ConstantPolicy(),
        rollout_policy=ConstantPolicy(),
        pre_switch_steps=2,
        min_post_switch_steps=3,
    )

    assert dataset.metadata["min_post_switch_steps"] == 3
    assert dataset.metadata["max_post_switch_age"] == 2


def test_collect_distillation_dataset_from_env_attaches_role_label() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    dataset = collect_distillation_dataset_from_env(
        _FakeDistillEnv(),
        num_samples=3,
        expected_student_obs_dim=7,
        expected_teacher_obs_dim=8,
        teacher_obs_key="obs",
        student_projection="drop_index",
        student_drop_index=3,
        action_mode="zero",
        role_label="walk_flat",
    )

    assert dataset.role_labels == ("walk_flat", "walk_flat", "walk_flat")
    assert dataset.metadata["role_label"] == "walk_flat"


def test_collect_distillation_dataset_from_env_pads_teacher_obs_tail() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    dataset = collect_distillation_dataset_from_env(
        _FakeDistillEnv(),
        num_samples=1,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=10,
        teacher_obs_key="obs",
        teacher_projection="pad_zeros",
        student_projection="identity",
        action_mode="zero",
    )

    assert dataset.student_obs.shape == (1, 8)
    assert dataset.teacher_obs.shape == (1, 10)
    assert torch.equal(dataset.teacher_obs[0, :8], torch.arange(8, dtype=torch.float32))
    assert torch.equal(dataset.teacher_obs[0, 8:], torch.zeros(2))
    assert dataset.metadata["teacher_projection"] == "pad_zeros"
    assert dataset.metadata["synthetic_teacher_tail"] is True


def test_collect_distillation_dataset_from_env_random_action_mode_is_nonzero() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    env = _FakeDistillEnv()
    dataset = collect_distillation_dataset_from_env(
        env,
        num_samples=3,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        teacher_obs_key="obs",
        student_projection="identity",
        action_mode="random",
        action_seed=7,
    )

    assert dataset.metadata["action_mode"] == "random"
    assert dataset.metadata["action_seed"] == 7
    assert dataset.metadata["action_abs_max"] > 0.0
    assert env.last_actions is not None
    assert np.isfinite(env.last_actions).all()
    assert np.max(np.abs(env.last_actions)) > 0.0
    assert dataset.teacher_actions is None


def test_collect_distillation_dataset_from_env_teacher_policy_action_mode() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    class FakeTeacherPolicy(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            assert obs.shape == (2, 8)
            return torch.tanh(obs[:, :3] * 0.01 + 0.1)

    env = _FakeDistillEnv()
    dataset = collect_distillation_dataset_from_env(
        env,
        num_samples=3,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        teacher_obs_key="obs",
        student_projection="identity",
        action_mode="teacher_policy",
        teacher_policy=FakeTeacherPolicy(),
    )

    assert dataset.metadata["action_mode"] == "teacher_policy"
    assert dataset.metadata["action_seed"] is None
    assert dataset.metadata["action_abs_max"] > 0.0
    assert env.last_actions is not None
    assert np.isfinite(env.last_actions).all()
    assert np.max(np.abs(env.last_actions)) > 0.0
    assert dataset.teacher_actions is not None
    assert dataset.teacher_actions.shape == (3, 3)
    assert torch.allclose(dataset.teacher_actions[:2], torch.as_tensor(env.last_actions))
    assert torch.isfinite(dataset.teacher_actions).all()


def test_collect_distillation_dataset_from_env_student_policy_rollout_mode() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    class FakeTeacherPolicy(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            assert obs.shape == (2, 8)
            return torch.full((obs.shape[0], 3), 0.25, dtype=obs.dtype, device=obs.device)

    class FakeRolloutPolicy(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            assert obs.shape == (2, 8)
            return torch.full((obs.shape[0], 3), -0.5, dtype=obs.dtype, device=obs.device)

    env = _FakeDistillEnv()
    dataset = collect_distillation_dataset_from_env(
        env,
        num_samples=3,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        teacher_obs_key="obs",
        student_projection="identity",
        action_mode="student_policy",
        teacher_policy=FakeTeacherPolicy(),
        rollout_policy=FakeRolloutPolicy(),
    )

    assert dataset.metadata["action_mode"] == "student_policy"
    assert dataset.metadata["action_seed"] is None
    assert dataset.metadata["rollout_policy"] == "distillation_student"
    assert dataset.metadata["action_abs_max"] == pytest.approx(0.5)
    assert env.last_actions is not None
    assert np.allclose(env.last_actions, -0.5)
    assert dataset.teacher_actions is not None
    assert dataset.teacher_actions.shape == (3, 3)
    assert torch.allclose(dataset.teacher_actions, torch.full((3, 3), 0.25))
    assert not torch.allclose(dataset.teacher_actions[:2], torch.as_tensor(env.last_actions))


def test_iterative_dagger_recollects_with_updated_student_policy() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        run_iterative_dagger_updates,
    )

    class ConstantPolicy(torch.nn.Module):
        def __init__(self, value: float, *, trainable: bool) -> None:
            super().__init__()
            bias = torch.full((3,), value, dtype=torch.float32)
            if trainable:
                self.bias = torch.nn.Parameter(bias)
            else:
                self.register_buffer("bias", bias)

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.bias.unsqueeze(0).expand(obs.shape[0], -1)

    class ActionHistoryEnv(_FakeDistillEnv):
        def __init__(self) -> None:
            super().__init__()
            self.action_history: list[np.ndarray] = []

        def step(self, actions):
            self.action_history.append(np.asarray(actions, dtype=np.float32).copy())
            return super().step(actions)

    student = ConstantPolicy(0.0, trainable=True)
    teacher = ConstantPolicy(0.5, trainable=False)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=torch.optim.SGD(student.parameters(), lr=0.5),
    )
    env = ActionHistoryEnv()

    result = run_iterative_dagger_updates(
        env,
        trainer=trainer,
        num_iterations=2,
        samples_per_iteration=4,
        batch_size=4,
        updates_per_iteration=1,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
    )

    assert result.iteration_count == 2
    assert result.update_count == 2
    assert result.samples_collected == 8
    assert [
        metadata["dagger_aggregate_num_samples"] for metadata in result.collection_metadata
    ] == [
        4,
        8,
    ]
    assert len(env.action_history) == 2
    assert np.allclose(env.action_history[0], 0.0)
    assert np.max(np.abs(env.action_history[1])) > 0.0


def test_dagger_rollout_uses_command_intent_expert_instead_of_soft_mixture() -> None:
    from unilab.algos.torch.distill import BehaviorDistillationTrainer, MoEStudentPolicy
    from unilab.algos.torch.distill.dagger import _resolve_dagger_rollout_policy

    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=2,
        num_experts=2,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        squash_action=False,
    )
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.zero_()
        student.experts[0].net[-1].bias.fill_(0.25)
        student.experts[1].net[-1].bias.fill_(-0.4)
        student.router[-1].bias.copy_(torch.tensor([-5.0, 5.0]))
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Linear(4, 2),
        optimizer=torch.optim.Adam(student.parameters()),
        command_intent_expert_targets={"active": 0, "inactive": 1},
        expert_behavior_loss_source="command_intent",
    )

    rollout_policy, expert_index, source = _resolve_dagger_rollout_policy(
        trainer,
        command_sample_filter="active",
        role_label="walk_flat",
    )

    assert expert_index == 0
    assert source == "command_intent"
    assert torch.allclose(rollout_policy(torch.zeros(1, 4)), torch.full((1, 2), 0.25))


def test_moe_expert_optimizer_state_does_not_drift_inactive_expert() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        DistillationBatch,
        MoEStudentPolicy,
    )

    student = MoEStudentPolicy(obs_dim=4, action_dim=2, num_experts=3, expert_hidden_dims=(8,))
    teacher = torch.nn.Linear(4, 2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=torch.optim.Adam(student.parameters(), lr=1.0e-2),
        command_intent_expert_targets={"active": 0, "inactive": 1},
        expert_behavior_loss_source="command_intent",
    )
    obs = torch.randn(8, 4)

    trainer.update(
        DistillationBatch(
            student_obs=obs,
            teacher_obs=obs,
            teacher_actions=torch.full((8, 2), 0.75),
            command_intents=("inactive",) * 8,
        )
    )
    stand_before = {
        key: value.detach().clone() for key, value in student.experts[1].state_dict().items()
    }

    for _ in range(5):
        trainer.update(
            DistillationBatch(
                student_obs=obs,
                teacher_obs=obs,
                teacher_actions=torch.full((8, 2), -0.75),
                command_intents=("active",) * 8,
            )
        )

    for key, value in student.experts[1].state_dict().items():
        assert torch.equal(value, stand_before[key]), key


def test_iterative_dagger_moves_collected_dataset_to_student_device(monkeypatch) -> None:
    from types import SimpleNamespace

    import unilab.algos.torch.distill.dagger as dagger_module
    from unilab.algos.torch.distill import build_distillation_dataset

    collected = build_distillation_dataset(
        torch.zeros(2, 8),
        torch.ones(2, 8),
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        expected_teacher_action_dim=3,
        teacher_actions=torch.zeros(2, 3),
    )
    student = torch.nn.Linear(8, 3, device="meta")
    trainer = SimpleNamespace(
        student=student,
        teacher=torch.nn.Linear(8, 3),
        update_count=1,
    )
    captured = {}

    monkeypatch.setattr(
        dagger_module,
        "collect_distillation_dataset_from_env",
        lambda *args, **kwargs: collected,
    )

    def fake_offline(trainer, dataset, **kwargs):
        captured["device"] = dataset.student_obs.device.type
        return SimpleNamespace(samples_seen=dataset.num_samples)

    monkeypatch.setattr(dagger_module, "run_offline_distillation_updates", fake_offline)

    dagger_module.run_iterative_dagger_updates(
        object(),
        trainer=trainer,
        num_iterations=1,
        samples_per_iteration=2,
        batch_size=2,
        updates_per_iteration=1,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
    )

    assert captured["device"] == "meta"


def test_collect_distillation_dataset_from_env_student_policy_resets_done_rows() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    class FakeTeacherPolicy(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return torch.full((obs.shape[0], 3), 0.25, dtype=obs.dtype, device=obs.device)

    class FakeRolloutPolicy(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return torch.full((obs.shape[0], 3), -0.5, dtype=obs.dtype, device=obs.device)

    class DoneAfterStepEnv(_FakeDistillEnv):
        def reset(self, env_indices):
            self.reset_calls += 1
            env_indices = np.asarray(env_indices, dtype=np.int32)
            base = np.arange(16, dtype=np.float32).reshape(2, 8)
            if env_indices.shape[0] == self.num_envs:
                rows = base
            else:
                rows = base[env_indices] + 100.0
            return {"obs": rows, "critic": rows + 100.0}, {
                "reset_indices": env_indices,
            }

        def step(self, actions):
            self.step_calls += 1
            self.last_actions = np.asarray(actions, dtype=np.float32)
            return type(
                "State",
                (),
                {
                    "obs": self._obs(self.step_calls),
                    "info": {},
                    "terminated": np.asarray([True, False], dtype=np.bool_),
                    "truncated": np.asarray([False, False], dtype=np.bool_),
                },
            )()

    env = DoneAfterStepEnv()
    dataset = collect_distillation_dataset_from_env(
        env,
        num_samples=4,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        teacher_obs_key="obs",
        student_projection="identity",
        action_mode="student_policy",
        teacher_policy=FakeTeacherPolicy(),
        rollout_policy=FakeRolloutPolicy(),
    )

    assert env.step_calls == 1
    assert env.reset_calls == 2
    assert dataset.metadata["action_mode"] == "student_policy"
    assert dataset.metadata["done_seen_samples"] == 1
    assert dataset.metadata["autoreset_done_count"] == 0
    assert dataset.metadata["manual_done_reset_count"] == 1
    assert torch.equal(dataset.student_obs[0], torch.arange(8, dtype=torch.float32))
    assert torch.equal(dataset.student_obs[1], torch.arange(8, 16, dtype=torch.float32))
    assert torch.equal(dataset.student_obs[2], torch.arange(8, dtype=torch.float32) + 100.0)
    assert torch.equal(dataset.student_obs[3], torch.arange(8, 16, dtype=torch.float32) + 1.0)


def test_collect_distillation_dataset_from_env_filters_active_command_samples() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    env = _CommandInfoDistillEnv(
        [
            np.asarray([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 0.0, 0.10], [0.0, 0.0, 0.0]], dtype=np.float32),
        ]
    )
    dataset = collect_distillation_dataset_from_env(
        env,
        num_samples=2,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        command_sample_filter="active",
        command_xy_threshold=0.05,
        command_yaw_threshold=0.05,
        max_env_steps=1,
    )

    assert dataset.num_samples == 2
    assert dataset.metadata["command_sample_filter"] == "active"
    assert dataset.metadata["command_seen_samples"] == 4
    assert dataset.metadata["command_selected_samples"] == 2
    assert dataset.metadata["env_steps"] == 1
    assert dataset.commands is not None
    assert torch.equal(
        dataset.commands,
        torch.tensor([[0.10, 0.0, 0.0], [0.0, 0.0, 0.10]], dtype=torch.float32),
    )
    assert dataset.command_intents == ("active", "active")
    assert dataset.metadata["command_intent_counts"] == {"active": 2}
    assert torch.equal(dataset.teacher_obs[0], torch.arange(8, 16, dtype=torch.float32))
    assert torch.equal(dataset.teacher_obs[1], torch.arange(8, dtype=torch.float32) + 1.0)


def test_collect_distillation_dataset_from_env_filters_inactive_command_samples() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    env = _CommandInfoDistillEnv(
        [
            np.asarray([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 0.0, 0.10], [0.0, 0.0, 0.0]], dtype=np.float32),
        ]
    )
    dataset = collect_distillation_dataset_from_env(
        env,
        num_samples=2,
        expected_student_obs_dim=8,
        expected_teacher_obs_dim=8,
        command_sample_filter="inactive",
        command_xy_threshold=0.05,
        command_yaw_threshold=0.05,
        max_env_steps=1,
    )

    assert dataset.num_samples == 2
    assert dataset.metadata["command_sample_filter"] == "inactive"
    assert dataset.metadata["command_seen_samples"] == 4
    assert dataset.metadata["command_selected_samples"] == 2
    assert dataset.metadata["env_steps"] == 1
    assert dataset.commands is not None
    assert torch.equal(
        dataset.commands,
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    assert dataset.command_intents == ("inactive", "inactive")
    assert dataset.metadata["command_intent_counts"] == {"inactive": 2}
    assert torch.equal(dataset.teacher_obs[0], torch.arange(8, dtype=torch.float32))
    assert torch.equal(dataset.teacher_obs[1], torch.arange(8, 16, dtype=torch.float32) + 1.0)


def test_collect_distillation_dataset_from_env_filter_requires_command_info() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    with pytest.raises(KeyError, match="commands"):
        collect_distillation_dataset_from_env(
            _FakeDistillEnv(),
            num_samples=1,
            expected_student_obs_dim=8,
            expected_teacher_obs_dim=8,
            command_sample_filter="active",
        )


def test_collect_distillation_dataset_from_env_filter_fails_when_budget_exhausts() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    env = _CommandInfoDistillEnv(
        [
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
        ]
    )
    with pytest.raises(RuntimeError, match="command_sample_filter='active'"):
        collect_distillation_dataset_from_env(
            env,
            num_samples=1,
            expected_student_obs_dim=8,
            expected_teacher_obs_dim=8,
            command_sample_filter="active",
            max_env_steps=1,
        )


def test_collect_distillation_dataset_from_env_rejects_half_open_projection() -> None:
    from unilab.algos.torch.distill import collect_distillation_dataset_from_env

    with pytest.raises(ValueError, match="student_drop_index"):
        collect_distillation_dataset_from_env(
            _FakeDistillEnv(),
            num_samples=1,
            expected_student_obs_dim=7,
            expected_teacher_obs_dim=8,
            student_projection="drop_index",
            student_drop_index=None,
        )


def test_offline_distillation_run_updates_and_saves_checkpoint(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        load_distillation_checkpoint,
        run_offline_distillation_updates,
    )

    torch.manual_seed(23)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    teacher = torch.nn.Linear(7, 3)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        loss_type="mse",
    )
    dataset = build_distillation_dataset(
        torch.randn(4, 5),
        torch.randn(4, 7),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
    )
    checkpoint_path = tmp_path / "offline_student.pt"

    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=2,
        max_updates=2,
        checkpoint_path=checkpoint_path,
        teacher_metadata={"algo": "linear-test"},
        distill_runtime_cfg={"loss_type": "mse"},
    )

    assert result.update_count == 2
    assert result.samples_seen == 4
    assert result.checkpoint_path == checkpoint_path
    assert result.last_loss >= 0.0
    assert result.last_behavior_loss == pytest.approx(result.last_loss)
    assert result.last_aux_loss == pytest.approx(0.0)
    assert result.last_expert_usage is None
    assert result.last_route_entropy is None
    assert result.last_teacher_action_source == "teacher"
    assert result.last_student_grad_norm > 0.0
    assert result.student_action_shape == (2, 3)
    assert result.teacher_action_shape == (2, 3)
    assert checkpoint_path.exists()

    restored = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    checkpoint = load_distillation_checkpoint(restored, checkpoint_path)
    assert checkpoint["agent_steps"] == 4
    assert checkpoint["teacher_metadata"] == {"algo": "linear-test"}
    assert checkpoint["distill_runtime_cfg"] == {"loss_type": "mse"}
    assert "optimizer_state_dict" in checkpoint
    for trained_param, restored_param in zip(student.parameters(), restored.parameters()):
        assert torch.allclose(trained_param, restored_param)


def test_offline_runtime_trace_emits_exact_failed_update_context(monkeypatch, capsys) -> None:
    monkeypatch.setenv("UNILAB_DISTILL_RUNTIME_DEBUG", "1")
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MoEStudentPolicy,
        build_distillation_dataset,
        run_offline_distillation_updates,
    )

    student = MoEStudentPolicy(
        obs_dim=2,
        action_dim=2,
        num_experts=2,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        squash_action=False,
    )
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Identity(),
        optimizer=torch.optim.Adam(student.parameters(), lr=0.0),
        command_intent_loss_coef=1.0,
        command_intent_expert_targets={"inactive": 0},
    )
    dataset = build_distillation_dataset(
        torch.zeros(2, 2),
        torch.empty(2, 0),
        expected_student_obs_dim=2,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=2,
        teacher_actions=torch.zeros(2, 2),
        command_intents=("active", "active"),
    )

    with pytest.raises(ValueError, match="unmapped command intent"):
        run_offline_distillation_updates(
            trainer,
            dataset,
            batch_size=2,
            max_updates=1,
        )

    prefix = "[distill-offline-runtime] "
    snapshots = [
        ast.literal_eval(line.removeprefix(prefix))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(prefix)
    ]
    assert [snapshot["stage"] for snapshot in snapshots] == [
        "offline/before_trainer_update",
        "offline/trainer_update_failure",
    ]
    failure = snapshots[-1]
    assert failure["update_number"] == 1
    assert failure["max_updates"] == 1
    assert failure["trainer_update_count"] == 0
    assert failure["error_type"] == "ValueError"
    assert failure["command_intent_counts"] == {"active": 2}
    assert failure["student_obs_shape"] == (2, 2)
    assert failure["recent_updates"] == [
        {
            "command_intent_counts": {"active": 2},
            "role_label_counts": {},
            "student_obs_shape": (2, 2),
            "teacher_obs_shape": (2, 0),
            "update_number": 1,
        }
    ]


def test_offline_distillation_run_accepts_cached_teacher_actions(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        run_offline_distillation_updates,
    )

    class RaisingTeacher(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            del obs
            raise AssertionError("offline cached teacher_action path must not call teacher")

    torch.manual_seed(29)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=RaisingTeacher(),
        optimizer=optimizer,
    )
    dataset = build_distillation_dataset(
        torch.randn(4, 5),
        torch.empty(4, 0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
        teacher_actions=torch.randn(4, 3),
        role_labels=("stand", "stand", "walk_height", "walk_height"),
    )

    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=2,
        max_updates=2,
        checkpoint_path=tmp_path / "cached_model.pt",
    )

    assert result.update_count == 2
    assert result.samples_seen == 4
    assert result.teacher_action_requires_grad is False
    assert result.teacher_action_shape == (2, 3)
    assert result.last_teacher_action_source == "cached"
    assert result.last_student_grad_norm > 0.0


def test_offline_distillation_run_can_repeat_dataset_for_multiple_updates() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        run_offline_distillation_updates,
    )

    torch.manual_seed(31)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Linear(7, 3),
        optimizer=optimizer,
    )
    dataset = build_distillation_dataset(
        torch.randn(3, 5),
        torch.randn(3, 7),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=7,
        role_labels=("walk", "stand", "walk"),
    )

    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=2,
        max_updates=4,
        repeat_dataset=True,
        shuffle=True,
        seed=5,
    )

    assert result.update_count == 4
    assert result.samples_seen == 6
    assert len(result.losses) == 4
    assert result.last_student_grad_norm > 0.0


def test_offline_distillation_run_balances_role_batches() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        run_offline_distillation_updates,
    )

    class RaisingTeacher(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            del obs
            raise AssertionError("balanced cached-target path must not call teacher")

    torch.manual_seed(37)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=RaisingTeacher(),
        optimizer=optimizer,
    )
    dataset = build_distillation_dataset(
        torch.randn(6, 5),
        torch.empty(6, 0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
        teacher_actions=torch.randn(6, 3),
        role_labels=("stand", "walk", "walk", "walk", "walk", "walk"),
    )

    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=4,
        max_updates=3,
        balance_key="role",
        balanced_labels=("stand", "walk"),
        seed=11,
    )

    assert result.update_count == 3
    assert result.samples_seen == 12
    assert result.batch_label_counts == (
        {"stand": 2, "walk": 2},
        {"stand": 2, "walk": 2},
        {"stand": 2, "walk": 2},
    )
    assert result.last_balance_label_counts == {"stand": 2, "walk": 2}
    assert result.last_teacher_action_source == "cached"
    assert result.last_student_grad_norm > 0.0


def test_balanced_label_pool_cache_is_immutable_bounded_and_rng_equivalent() -> None:
    from unilab.algos.torch.distill.offline import (
        BalancedLabelIndexPools,
        _build_balanced_label_pools,
        _sample_balanced_batch_indices_from_pools,
    )

    labels = ("walk", "stand", "walk", "transition", "stand", "walk")
    selected = ("walk", "stand", "transition")
    cached = _build_balanced_label_pools(labels, selected, balance_key="scenario")

    assert isinstance(cached, BalancedLabelIndexPools)
    assert cached.source_labels is labels
    assert cached.balance_key == "scenario"
    assert cached.selected_labels == selected
    assert all(indices.device.type == "cpu" for indices in cached.row_indices)
    assert all(indices.dtype == torch.int64 for indices in cached.row_indices)
    assert all(indices.is_contiguous() for indices in cached.row_indices)
    assert cached.payload_bytes == 8 * len(labels)
    assert cached.payload_bytes <= 8 * len(labels)

    rebuilt_generator = torch.Generator().manual_seed(23)
    cached_generator = torch.Generator().manual_seed(23)
    for _ in range(5):
        rebuilt = _build_balanced_label_pools(labels, selected, balance_key="scenario")
        rebuilt_indices, rebuilt_counts = _sample_balanced_batch_indices_from_pools(
            rebuilt,
            batch_size=6,
            balance_quotas={"walk": 0.5, "stand": 0.25, "transition": 0.25},
            generator=rebuilt_generator,
        )
        cached_indices, cached_counts = _sample_balanced_batch_indices_from_pools(
            cached,
            batch_size=6,
            balance_quotas={"walk": 0.5, "stand": 0.25, "transition": 0.25},
            generator=cached_generator,
        )
        assert torch.equal(rebuilt_indices, cached_indices)
        assert rebuilt_counts == cached_counts
    assert torch.equal(rebuilt_generator.get_state(), cached_generator.get_state())

    with pytest.raises(ValueError, match="does not match source labels"):
        BalancedLabelIndexPools(
            source_labels=labels,
            balance_key="scenario",
            selected_labels=selected,
            row_indices=(
                torch.tensor([0, 1, 2]),
                torch.tensor([1, 4]),
                torch.tensor([3]),
            ),
        )


def test_offline_distillation_builds_balanced_label_pools_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        offline,
        run_offline_distillation_updates,
    )

    dataset = build_distillation_dataset(
        torch.randn(6, 5),
        torch.empty(6, 0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
        teacher_actions=torch.randn(6, 3),
        role_labels=("stand", "walk", "walk", "walk", "walk", "walk"),
    )
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Linear(0, 3),
        optimizer=torch.optim.Adam(student.parameters(), lr=1e-2),
    )
    original = offline._build_balanced_label_pools
    build_count = 0

    def counted_build(*args: object, **kwargs: object) -> object:
        nonlocal build_count
        build_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(offline, "_build_balanced_label_pools", counted_build)

    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=4,
        max_updates=3,
        balance_key="role",
        balanced_labels=("stand", "walk"),
        seed=11,
    )

    assert result.update_count == 3
    assert build_count == 1


def test_offline_distillation_run_balances_command_intent_batches() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        run_offline_distillation_updates,
    )

    torch.manual_seed(41)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Linear(0, 3),
        optimizer=torch.optim.Adam(student.parameters(), lr=1e-2),
    )
    dataset = build_distillation_dataset(
        torch.randn(6, 5),
        torch.empty(6, 0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
        teacher_actions=torch.randn(6, 3),
        command_intents=("inactive", "active", "active", "active", "active", "active"),
    )

    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=4,
        max_updates=2,
        balance_key="command_intent",
        balanced_labels=("inactive", "active"),
        seed=13,
    )

    assert result.batch_label_counts == (
        {"inactive": 2, "active": 2},
        {"inactive": 2, "active": 2},
    )
    assert result.last_balance_label_counts == {"inactive": 2, "active": 2}
    assert result.samples_seen == 8


def test_offline_distillation_run_applies_scenario_quotas() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        run_offline_distillation_updates,
    )

    torch.manual_seed(43)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Linear(0, 3),
        optimizer=torch.optim.Adam(student.parameters(), lr=1e-2),
    )
    scenario_labels = ("walk_flat",) * 10 + ("static_stand",) * 10 + ("walk_to_stop",) * 10
    role_labels = ("walk_flat",) * 10 + ("stand",) * 20
    command_intents = ("active",) * 10 + ("inactive",) * 20
    transition_ages = torch.full((30,), -1, dtype=torch.int64)
    transition_ages[20:] = 0
    command_before = torch.zeros(30, 3)
    command_before[:10] = 0.4
    command_after = torch.zeros(30, 3)
    dataset = build_distillation_dataset(
        torch.randn(30, 5),
        torch.empty(30, 0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
        teacher_actions=torch.randn(30, 3),
        role_labels=role_labels,
        command_intents=command_intents,
        scenario_labels=scenario_labels,
        transition_ages=transition_ages,
        command_before=command_before,
        command_after=command_after,
    )

    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=20,
        max_updates=1,
        balance_key="scenario",
        balanced_labels=("walk_flat", "static_stand", "walk_to_stop"),
        balance_quotas={"walk_flat": 0.5, "static_stand": 0.25, "walk_to_stop": 0.25},
        seed=17,
    )

    assert result.batch_label_counts == ({"walk_flat": 10, "static_stand": 5, "walk_to_stop": 5},)
    assert result.last_balance_label_counts == {
        "walk_flat": 10,
        "static_stand": 5,
        "walk_to_stop": 5,
    }


def test_offline_distillation_run_enforces_transition_replay_budget() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        run_offline_distillation_updates,
    )

    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Linear(0, 3),
        optimizer=torch.optim.Adam(student.parameters(), lr=1e-2),
    )
    dataset = build_distillation_dataset(
        torch.randn(10, 5),
        torch.empty(10, 0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
        teacher_actions=torch.randn(10, 3),
        role_labels=("walk_flat",) * 4 + ("stand",) * 4 + ("stand",) * 2,
        command_intents=("active",) * 4 + ("inactive",) * 6,
        scenario_labels=("walk_flat",) * 4 + ("static_stand",) * 4 + ("walk_to_stop",) * 2,
        transition_ages=torch.tensor([-1] * 8 + [0, 1], dtype=torch.int64),
        command_before=torch.tensor([[0.4, 0.0, 0.0]] * 4 + [[0.0, 0.0, 0.0]] * 6),
        command_after=torch.zeros(10, 3),
    )

    with pytest.raises(ValueError, match="required_updates=3"):
        run_offline_distillation_updates(
            trainer,
            dataset,
            batch_size=10,
            max_updates=2,
            balance_key="scenario",
            balanced_labels=("walk_flat", "static_stand", "walk_to_stop"),
            balance_quotas={
                "walk_flat": 0.4,
                "static_stand": 0.4,
                "walk_to_stop": 0.2,
            },
            min_balanced_replay_passes=3,
            min_balanced_replay_labels=("walk_to_stop",),
        )


def test_offline_distillation_run_balanced_sampler_fails_closed() -> None:
    from unilab.algos.torch.distill import (
        BehaviorDistillationTrainer,
        MLPStudentPolicy,
        build_distillation_dataset,
        run_offline_distillation_updates,
    )

    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Linear(0, 3),
        optimizer=torch.optim.Adam(student.parameters(), lr=1e-2),
    )
    dataset = build_distillation_dataset(
        torch.randn(2, 5),
        torch.empty(2, 0),
        expected_student_obs_dim=5,
        expected_teacher_obs_dim=0,
        expected_teacher_action_dim=3,
        teacher_actions=torch.randn(2, 3),
    )

    with pytest.raises(ValueError, match="role_labels"):
        run_offline_distillation_updates(
            trainer,
            dataset,
            batch_size=2,
            max_updates=1,
            balance_key="role",
        )


def test_distillation_student_checkpoint_loads_for_student_only_playback(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        MLPStudentPolicy,
        load_distillation_student_policy,
        save_distillation_checkpoint,
    )

    torch.manual_seed(29)
    student = MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
    checkpoint_path = tmp_path / "student_play.pt"
    save_distillation_checkpoint(
        checkpoint_path,
        student=student,
        agent_steps=4,
        teacher_metadata={"task_name": "G1WalkHeight"},
        distill_runtime_cfg={
            "student_obs_dim": 5,
            "student_action_dim": 3,
            "student_hidden_dims": [8],
            "student_activation": "elu",
            "student_squash_action": True,
        },
    )

    loaded = load_distillation_student_policy(checkpoint_path, device="cpu")
    obs = torch.randn(2, 5)
    action = loaded.policy(obs)

    assert loaded.obs_dim == 5
    assert loaded.action_dim == 3
    assert loaded.agent_steps == 4
    assert loaded.teacher_metadata == {"task_name": "G1WalkHeight"}
    assert action.shape == (2, 3)
    assert action.requires_grad is False
    assert torch.isfinite(action).all()

    with pytest.raises(ValueError, match="Student obs dim mismatch"):
        loaded.policy(torch.randn(2, 6))


def test_distillation_moe_student_checkpoint_loads_for_student_only_playback(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        MoEStudentPolicy,
        load_distillation_student_policy,
        save_distillation_checkpoint,
    )

    torch.manual_seed(31)
    student = MoEStudentPolicy(
        obs_dim=5,
        action_dim=3,
        num_experts=3,
        expert_hidden_dims=(8,),
        router_hidden_dims=(4,),
        routing_mode="soft",
        router_temperature=0.75,
        squash_action=False,
    )
    checkpoint_path = tmp_path / "moe_student_play.pt"
    save_distillation_checkpoint(
        checkpoint_path,
        student=student,
        agent_steps=7,
        teacher_metadata={"task_name": "G1WalkHeight", "student": "moe"},
        distill_runtime_cfg={
            "student_model_type": "moe",
            "student_obs_dim": 5,
            "student_action_dim": 3,
            "student_num_experts": 3,
            "student_expert_hidden_dims": [8],
            "student_router_hidden_dims": [4],
            "student_routing_mode": "soft",
            "student_router_temperature": 0.75,
            "student_activation": "elu",
            "student_squash_action": False,
        },
    )

    loaded = load_distillation_student_policy(checkpoint_path, device="cpu")
    obs = torch.randn(2, 5)
    action = loaded.policy(obs)

    assert isinstance(loaded.policy, MoEStudentPolicy)
    assert loaded.obs_dim == 5
    assert loaded.action_dim == 3
    assert loaded.agent_steps == 7
    assert loaded.teacher_metadata == {"task_name": "G1WalkHeight", "student": "moe"}
    assert loaded.distill_runtime_cfg["student_model_type"] == "moe"
    assert loaded.policy.num_experts == 3
    assert loaded.policy.router_temperature == pytest.approx(0.75)
    assert action.shape == (2, 3)
    assert action.requires_grad is False
    assert torch.isfinite(action).all()
    assert all(param.requires_grad is False for param in loaded.policy.parameters())

    with pytest.raises(ValueError, match="Student obs dim mismatch"):
        loaded.policy(torch.randn(2, 6))


def test_distillation_student_playback_rejects_unknown_model_type(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        MLPStudentPolicy,
        load_distillation_student_policy,
        save_distillation_checkpoint,
    )

    checkpoint_path = tmp_path / "student_unknown_model.pt"
    save_distillation_checkpoint(
        checkpoint_path,
        student=MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,)),
        agent_steps=4,
        distill_runtime_cfg={
            "student_model_type": "unknown",
            "student_obs_dim": 5,
            "student_action_dim": 3,
        },
    )

    with pytest.raises(ValueError, match="student_model_type"):
        load_distillation_student_policy(checkpoint_path, device="cpu")


def test_distillation_student_playback_rejects_missing_runtime_dims(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        MLPStudentPolicy,
        load_distillation_student_policy,
        save_distillation_checkpoint,
    )

    checkpoint_path = tmp_path / "student_missing_dims.pt"
    save_distillation_checkpoint(
        checkpoint_path,
        student=MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,)),
        agent_steps=4,
        distill_runtime_cfg={"loss_type": "mse"},
    )

    with pytest.raises(ValueError, match="student_obs_dim"):
        load_distillation_student_policy(checkpoint_path, device="cpu")


def test_sac_teacher_checkpoint_loads_with_dim_guard(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        DistillationTeacherSpec,
        load_sac_teacher_policy,
    )
    from unilab.algos.torch.fast_sac.learner import SACActor

    torch.manual_seed(17)
    actor = SACActor(
        obs_dim=5,
        action_dim=3,
        hidden_dim=8,
        use_layer_norm=False,
        device="cpu",
    )
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save({"actor": actor.state_dict(), "update_count": 3}, checkpoint_path)

    teacher = load_sac_teacher_policy(
        checkpoint_path,
        DistillationTeacherSpec(
            algo_type="sac",
            obs_dim=5,
            action_dim=3,
            actor_hidden_dim=8,
            use_layer_norm=False,
        ),
    )
    action = teacher(torch.randn(4, 5))

    assert action.shape == (4, 3)
    assert action.requires_grad is False
    assert torch.isfinite(action).all()
    assert all(param.requires_grad is False for param in teacher.parameters())


def test_sac_teacher_checkpoint_inspector_reports_actor_input_dim(tmp_path) -> None:
    from unilab.algos.torch.distill import inspect_sac_teacher_checkpoint
    from unilab.algos.torch.fast_sac.learner import SACActor

    actor = SACActor(
        obs_dim=5,
        action_dim=3,
        hidden_dim=8,
        use_layer_norm=False,
        device="cpu",
    )
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save({"actor": actor.state_dict()}, checkpoint_path)

    info = inspect_sac_teacher_checkpoint(checkpoint_path)

    assert info.checkpoint_path == str(checkpoint_path)
    assert info.actor_input_dim == 5
    assert info.first_weight_key == "net.0.weight"


def test_sac_teacher_checkpoint_rejects_dim_mismatch(tmp_path) -> None:
    from unilab.algos.torch.distill import (
        DistillationTeacherSpec,
        load_sac_teacher_policy,
    )
    from unilab.algos.torch.fast_sac.learner import SACActor

    actor = SACActor(
        obs_dim=5,
        action_dim=3,
        hidden_dim=8,
        use_layer_norm=False,
        device="cpu",
    )
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save({"actor": actor.state_dict()}, checkpoint_path)

    with pytest.raises(ValueError, match="checkpoint actor input dim=5"):
        load_sac_teacher_policy(
            checkpoint_path,
            DistillationTeacherSpec(
                algo_type="sac",
                obs_dim=6,
                action_dim=3,
                actor_hidden_dim=8,
                use_layer_norm=False,
            ),
        )
