"""Joint NTASAD sampler/replay/denoiser unrolling with explicit method policies."""

from dataclasses import dataclass

import torch
from torch import nn

from .adaptive import AdaptiveSampler
from .adaptive_model import AdaptiveNTASADDenoiser, DEFAULT_RECONSTRUCTION
from .allocation import POLICIES, allocate_samples
from .model import INPUT_CHANNELS
from .ntasad_radiance import decode_radiance
from .renderer_gradient import ZERO_POLICIES, differentiable_replay
from .sample_layers import MAX_SPP, replay_sample_layers


ARCHITECTURE = "ntasad_joint_v2"
LEGACY_ARCHITECTURE = "ntasad_joint_v1"
WARP_VERSION = "bilinear_gather_v1"


@dataclass(frozen=True)
class JointPrediction:
    radiance: torch.Tensor  # NT3HW linear, or transformed when requested.
    valid: torch.Tensor     # NT1HW availability; never used to hide loss pixels.
    expected: torch.Tensor  # NT1HW continuous expected allocations.
    counts: torch.Tensor    # NT1HW actual rounded counts, including initialization.


class JointNTASAD(nn.Module):
    def __init__(self, denoiser_channels=48, sampler_channels=48, average_spp=4,
                 allocation_policy="capped_softmax_v1", max_spp=MAX_SPP,
                 zero_policy="unit_denominator", reconstruction_version=DEFAULT_RECONSTRUCTION):
        super().__init__()
        if type(average_spp) is not int or type(max_spp) is not int or not 1 <= average_spp <= max_spp <= MAX_SPP:
            raise ValueError("Joint average_spp and max_spp must be integers with 1<=average_spp<=max_spp<=63")
        if allocation_policy not in POLICIES or zero_policy not in ZERO_POLICIES:
            raise ValueError("Unknown joint allocation or zero-count gradient policy")
        self.sampler = AdaptiveSampler(sampler_channels)
        self.denoiser = AdaptiveNTASADDenoiser(denoiser_channels,history_backend="gather",
                                             reconstruction_version=reconstruction_version)
        self.average_spp, self.max_spp = average_spp,max_spp
        self.allocation_policy, self.zero_policy = allocation_policy,zero_policy

    def specification(self):
        return {"denoiser_channels":self.denoiser.base_channels,"sampler_channels":self.sampler.base_channels,
                "average_spp":self.average_spp,"max_spp":self.max_spp,
                "allocation_policy":self.allocation_policy,"zero_policy":self.zero_policy,
                "reconstruction_version":self.denoiser.reconstruction_version}

    def forward_sequence(self, buffers, layers, *, training_reference=None, transformed=False) -> JointPrediction:
        """Initialize uniformly, then unroll both networks without detaching history.

        Inference needs guides and layers only. References affect backward of
        replay only; loss evaluation obtains its targets separately. In training
        mode with gradients enabled, omission of training_reference is an error.
        """
        missing = INPUT_CHANNELS.keys()-buffers.keys()
        if missing:
            raise ValueError(f"Missing joint buffers: {sorted(missing)}")
        rgb = buffers["radiance"]
        if rgb.ndim != 5 or min(rgb.shape) < 1 or rgb.shape[2] != 3:
            raise ValueError("Expected nonempty NT3HW joint buffers")
        n,t,_,h,w = rgb.shape
        if layers.shape != (n,t,6,3,h,w) or layers.device != rgb.device or layers.dtype != rgb.dtype:
            raise ValueError("Joint layers must be matching NT6x3HW")
        for key,channels in INPUT_CHANNELS.items():
            if buffers[key].shape != (n,t,channels,h,w):
                raise ValueError(f"Invalid joint sequence buffer {key}")
        if training_reference is None and self.training and torch.is_grad_enabled():
            raise ValueError("Joint training requires training_reference for allocation gradients")
        if training_reference is not None and (training_reference.shape != rgb.shape
                or training_reference.dtype != rgb.dtype or training_reference.device != rgb.device
                or not torch.isfinite(training_reference).all() or (training_reference < 0).any()):
            raise ValueError("Training reference must be matching finite nonnegative NT3HW")
        state = None
        outputs,validity,expected,counts = [],[],[],[]
        for index in range(t):
            frame = {key:buffers[key][:,index] for key in INPUT_CHANNELS}
            if index == 0:
                count = torch.full((n,1,h,w),self.average_spp,device=rgb.device,dtype=torch.int64)
                expectation = count.to(rgb.dtype)
                current = replay_sample_layers(layers[:,index],count)
            else:
                logits = self.sampler.predict_logits(frame,decode_radiance(state.radiance),
                                                      history_available=state.valid,history_backend="gather")
                allocation = allocate_samples(logits,self.average_spp,policy=self.allocation_policy,max_spp=self.max_spp)
                expectation,count = allocation.expected,allocation.counts
                if training_reference is None:
                    current = replay_sample_layers(layers[:,index],count)
                else:
                    render = differentiable_replay(layers[:,index],expectation,training_reference[:,index],
                                                   zero_policy=self.zero_policy)
                    current = render.radiance
            frame = dict(frame,radiance=current,sample_count=count)
            state = self.denoiser._step(frame,state)
            outputs.append(state.radiance if transformed else decode_radiance(state.radiance))
            validity.append(state.valid)
            expected.append(expectation)
            counts.append(count)
        return JointPrediction(torch.stack(outputs,1),torch.stack(validity,1),
                               torch.stack(expected,1),torch.stack(counts,1))
