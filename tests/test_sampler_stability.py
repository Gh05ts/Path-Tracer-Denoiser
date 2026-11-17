"""Regression coverage for the minimum-one policy used in collapse-prevention runs."""
import json

import pytest
import torch

from wavefront_denoiser.allocation import allocate_samples
from wavefront_denoiser.joint import JointNTASAD
from wavefront_denoiser.joint_checkpoint import joint_optimizer
from wavefront_denoiser.joint_training import training_config
from wavefront_denoiser.renderer_gradient import differentiable_replay
from wavefront_denoiser.sample_layers import replay_sample_layers
from wavefront_denoiser.training_defaults import joint_training_config


@pytest.mark.parametrize('spp,maximum', [(1, 1), (1, 63), (4, 63), (63, 63)])
def test_minimum_one_extreme_logits_keep_exact_independent_budgets(spp, maximum):
    logits = torch.tensor([[[[1e20, -1e20, -1e20, -1e20]]],
                           [[[.1, -.1, .2, -.2]]]], requires_grad=True)
    result = allocate_samples(logits, spp, policy='minimum_one_exact_v1', max_spp=maximum)
    assert (result.counts >= 1).all() and (result.counts <= maximum).all()
    assert torch.all(result.counts.flatten(1).sum(1) == spp*4)
    (result.expected * torch.arange(4)).sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_repaired_counts_drive_forward_and_surrogate_denominator():
    # Independent rounding would use [2, 2, 2], whereas the allocator repairs
    # the fractional tie deterministically to the exact budget [2, 2, 1].
    expected = torch.full((1, 1, 1, 3), 5/3, requires_grad=True)
    counts = torch.tensor([[[[2, 2, 1]]]])
    layers = torch.arange(1, 7, dtype=torch.float32).view(1, 6, 1, 1, 1).expand(1, 6, 3, 1, 3)
    target = torch.full((1, 3, 1, 3), 10.)
    render = differentiable_replay(layers, expected, target, zero_policy='unit_denominator', counts=counts)
    torch.testing.assert_close(render.radiance, replay_sample_layers(layers, counts), rtol=0, atol=0)
    render.radiance.sum().backward()
    oracle = ((target-render.radiance.detach())/counts).sum(1, keepdim=True)
    torch.testing.assert_close(expected.grad, oracle)


def test_new_profile_and_legacy_optimizer_settings(tmp_path):
    path = tmp_path/'train.json'
    path.write_text(json.dumps(joint_training_config()))
    stable = training_config(path)
    assert stable['allocation_policy'] == 'minimum_one_exact_v1'
    assert stable['radiance_gradient_floor'] == 1e-3
    model = JointNTASAD(4, 4)
    opt = joint_optimizer(model, stable['learning_rate'], sampler_learning_rate=stable['sampler_learning_rate'])
    assert [g['lr'] for g in opt.param_groups] == pytest.approx([1e-5, 1e-4])
    legacy = dict(joint_training_config(), allocation_policy='capped_softmax_v1', learning_rate=.001)
    path.write_text(json.dumps(legacy))
    parsed = training_config(path)
    assert 'sampler_learning_rate' not in parsed and 'radiance_gradient_floor' not in parsed
    assert [g['lr'] for g in joint_optimizer(model, .001).param_groups] == [.001, .001]


def test_fractional_budget_is_rejected_by_integer_policy():
    with pytest.raises(ValueError, match='integer'):
        allocate_samples(torch.zeros(1, 1, 2, 2), 1.5, policy='minimum_one_exact_v1')
