"""Separate two-network weights and epoch-boundary optimizer/RNG checkpoints."""

import os
import math
from pathlib import Path
import pickle
import tempfile

import torch

from .adaptive import ARCHITECTURE as SAMPLER_ARCHITECTURE
from .adaptive_model import (ARCHITECTURE as DENOISER_ARCHITECTURE,
                             LEGACY_ARCHITECTURE as LEGACY_DENOISER_ARCHITECTURE, FEATURE_VERSION)
from .checkpoint import (_cpu_tree, _json_metadata, _optimizer_parameters, _optimizer_type,
                         _rng_state, restore_training)
from .joint import ARCHITECTURE, LEGACY_ARCHITECTURE, WARP_VERSION, JointNTASAD
from .renderer_gradient import GRADIENT_VERSION
from .sample_layers import LAYER_SPP


CHECKPOINT_FORMAT = "wavefront-ntasad-joint-checkpoint"
DTYPES = {"float32":torch.float32,"float64":torch.float64}
IDENTITY = {"format":CHECKPOINT_FORMAT,"version":2,"architecture":ARCHITECTURE,
            "sampler_architecture":SAMPLER_ARCHITECTURE,"denoiser_architecture":DENOISER_ARCHITECTURE,
            "feature_version":FEATURE_VERSION,"gradient_version":GRADIENT_VERSION,"warp_version":WARP_VERSION,
            "layer_spp":list(LAYER_SPP)}
LEGACY_IDENTITY = dict(IDENTITY, version=1, architecture=LEGACY_ARCHITECTURE,
                       denoiser_architecture=LEGACY_DENOISER_ARCHITECTURE)
SPEC_KEYS = {"denoiser_channels","sampler_channels","average_spp","max_spp",
             "allocation_policy","zero_policy","reconstruction_version"}


def checkpoint_spec(payload):
    """Read original v1 semantics explicitly; never reinterpret old weights as v2."""
    spec = payload.get("model_spec")
    legacy = payload.get("version") == 1
    required = SPEC_KEYS - {"reconstruction_version"} if legacy else SPEC_KEYS
    if not isinstance(spec,dict) or set(spec) != required:
        raise ValueError("Missing joint model/policy specification")
    return dict(spec,reconstruction_version="sample_mass_v1") if legacy else dict(spec)


def joint_optimizer(model: JointNTASAD, learning_rate=.001):
    return torch.optim.Adam([{"params":model.sampler.parameters(),"name":"sampler"},
                             {"params":model.denoiser.parameters(),"name":"denoiser"}],lr=learning_rate)


def _check_config(spec, config):
    if any(key in config and config[key] != value for key,value in spec.items()):
        raise ValueError("Joint checkpoint config disagrees with its model/policies")


def _finite_state(value):
    if isinstance(value,torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value,dict):
        return all(_finite_state(v) for v in value.values())
    if isinstance(value,(tuple,list)):
        return all(_finite_state(v) for v in value)
    return math.isfinite(value) if isinstance(value,float) else True


def save_joint_checkpoint(path, model: JointNTASAD, *, optimizer=None, epoch=0, global_step=0,
                          config=None, dataset_split=None, generator=None):
    if not isinstance(model,JointNTASAD):
        raise ValueError("Expected JointNTASAD, not a uniform checkpoint")
    for value in (epoch,global_step):
        if type(value) is not int or value < 0:
            raise ValueError("Joint progress must use nonnegative integer epochs/steps")
    dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    if dtype not in DTYPES or any(p.dtype != DTYPES[dtype] for p in model.parameters()):
        raise ValueError("Joint weights must share float32 or float64 dtype")
    weights = _cpu_tree(model.state_dict())
    if any(not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError("Joint checkpoint weights must be finite")
    config,split = _json_metadata(config),_json_metadata(dataset_split)
    _check_config(model.specification(),config)
    if generator is not None and optimizer is None:
        raise ValueError("A loader generator requires resumable optimizer state")
    training = None
    if optimizer is not None:
        training = {"optimizer_type":_optimizer_type(optimizer),
                    "parameter_groups":_optimizer_parameters(model,optimizer),
                    "optimizer_state":_cpu_tree(optimizer.state_dict()),"rng":_cpu_tree(_rng_state(generator))}
        if not _finite_state(training["optimizer_state"]):
            raise ValueError("Joint optimizer state must be finite")
    payload = dict(IDENTITY,denoiser_architecture=model.denoiser.architecture,
        model_spec=model.specification(),parameter_dtype=dtype,state_dict=weights,
        epoch=epoch,global_step=global_step,config=config,dataset_split=split,training=training,
        torch_version=str(torch.__version__),training_device=str(next(model.parameters()).device),
        execution_state={"deterministic":torch.are_deterministic_algorithms_enabled(),
            "warn_only":torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_benchmark":torch.backends.cudnn.benchmark})
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary = tempfile.mkstemp(prefix=path.name+".",suffix=".partial",dir=path.parent)
    try:
        with os.fdopen(descriptor,"wb") as stream:
            torch.save(payload,stream)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_joint_checkpoint(path, device=torch.device("cpu")):
    try:
        payload = torch.load(path,map_location="cpu",weights_only=True)
    except (pickle.UnpicklingError,EOFError,RuntimeError,IndexError) as error:
        raise ValueError(f"Invalid joint checkpoint: {error}") from error
    if not isinstance(payload,dict):
        raise ValueError("Incompatible joint checkpoint format, architecture, gradient or sample bank")
    identity = LEGACY_IDENTITY if payload.get("version") == 1 else dict(IDENTITY)
    if payload.get("version") == 2:
        spec = checkpoint_spec(payload)
        if spec["reconstruction_version"] == "sample_mass_v1":
            identity["denoiser_architecture"] = LEGACY_DENOISER_ARCHITECTURE
    if any(payload.get(k) != v or type(payload.get(k)) is not type(v) for k,v in identity.items()):
        raise ValueError("Incompatible joint checkpoint format, architecture, gradient or sample bank")
    spec = checkpoint_spec(payload)
    for key in ("epoch","global_step"):
        if type(payload.get(key)) is not int or payload[key] < 0:
            raise ValueError(f"Invalid joint checkpoint {key}")
    for key in ("config","dataset_split"):
        if not isinstance(payload.get(key),dict):
            raise ValueError(f"Missing joint {key}")
        _json_metadata(payload[key])
    _check_config(spec,payload["config"])
    dtype = DTYPES.get(payload.get("parameter_dtype")) if isinstance(payload.get("parameter_dtype"),str) else None
    weights = payload.get("state_dict")
    if dtype is None or not isinstance(weights,dict) or any(not isinstance(v,torch.Tensor) or v.dtype != dtype
            or not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError("Invalid joint checkpoint dtype or finite weights")
    model = JointNTASAD(**spec).to(dtype=dtype)
    try:
        model.load_state_dict(weights,strict=True)
    except RuntimeError as error:
        raise ValueError(f"Incompatible joint weights: {error}") from error
    return model.to(device).eval(),payload


def restore_joint_training(model, optimizer, payload, *, generator=None, restore_rng=True):
    if checkpoint_spec(payload) != model.specification():
        raise ValueError("Joint restore model or policies differ")
    if restore_rng and payload.get("training_device") != str(next(model.parameters()).device):
        raise ValueError("Training device changed; use restore_rng=False for deliberate migration")
    training = payload.get("training")
    if isinstance(training,dict) and (not isinstance(training.get("optimizer_state"),dict)
                                      or not _finite_state(training["optimizer_state"])):
        raise ValueError("Invalid/non-finite joint optimizer state")
    flags = payload.get("execution_state")
    if not isinstance(flags,dict) or set(flags) != {"deterministic","warn_only","cudnn_deterministic","cudnn_benchmark"} or any(type(v) is not bool for v in flags.values()):
        raise ValueError("Missing/invalid joint execution flags")
    torch.use_deterministic_algorithms(flags["deterministic"],warn_only=flags["warn_only"])
    torch.backends.cudnn.deterministic = flags["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = flags["cudnn_benchmark"]
    return restore_training(model,optimizer,payload,generator=generator,restore_rng=restore_rng)
