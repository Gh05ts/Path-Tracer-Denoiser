"""Aligned v5 layer/guide crops and strict selected-partition inventories."""

from pathlib import Path
from contextlib import contextmanager
import fcntl

import numpy as np
import torch

from .adaptive_data import V5SequenceDataset, load_v5_frame, read_json, read_v5_dataset, verify_v5_record
from .dataset_generation import digest
from .frames import Frame, read_manifest
from .sequence_data import transform_window


class JointPatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, *, partition, length=5, patch_size=128, patches_per_window=1, augment=True,
                 sequence_ids=None):
        if (type(patch_size) is not int or patch_size < 1 or type(patches_per_window) is not int
                or patches_per_window < 1 or type(length) is not int or length < 3 or type(augment) is not bool):
            raise ValueError("Need positive joint crop/count and at least three frames")
        if not augment and patches_per_window > 5:
            raise ValueError("Validation supports one to five fixed crops")
        self.sequences = V5SequenceDataset(root,partition=partition,length=length,tensors=False,sequence_ids=sequence_ids)
        descriptor = read_v5_dataset(root)
        for group in (descriptor["partitions"][partition] if sequence_ids is None else sequence_ids):
            config = descriptor["groups"][group]["source_config"]
            if patch_size > min(config["width"],config["height"]):
                raise ValueError("Joint patch size exceeds source dimensions")
        self.patch_size,self.patches_per_window,self.augment = patch_size,patches_per_window,augment

    def __len__(self):
        return len(self.sequences)*self.patches_per_window

    def __getitem__(self,index):
        group,start,length = self.sequences.windows[index//self.patches_per_window]
        config = read_v5_dataset(self.sequences.root)["groups"][group]["source_config"]
        h,w,size = config["height"],config["width"],self.patch_size
        if self.augment:
            top,left = int(torch.randint(h-size+1,(1,))),int(torch.randint(w-size+1,(1,)))
        else:
            origins = [((h-size)//2,(w-size)//2),(0,0),(0,w-size),(h-size,0),(h-size,w-size)]
            top,left = origins[index%self.patches_per_window]
        values,layers = [],[]
        for frame_index in range(start,start+length):
            # Release each decoded full frame before decoding the next. Only
            # cropped copies survive into a five-frame training batch.
            frame = load_v5_frame(self.sequences.root,group,frame_index)
            cropped = {key:np.ascontiguousarray(value[top:top+size,left:left+size]) for key,value in frame.base.buffers.items()}
            values.append(Frame(frame.base.manifest_path,frame.base.metadata,cropped).as_tensors())
            layer = np.ascontiguousarray(frame.layers[:,top:top+size,left:left+size])
            layers.append(torch.from_numpy(layer).permute(0,3,1,2).contiguous())
            del frame,cropped,layer
        buffers = {key:torch.stack([v[key] for v in values]) for key in values[0]}
        buffers["sample_layers"] = torch.stack(layers).flatten(1,2)
        transformed = transform_window(buffers,top=0,left=0,size=size,
            flip_x=bool(torch.randint(2,(1,))) if self.augment else False,
            flip_y=bool(torch.randint(2,(1,))) if self.augment else False,
            rotations=int(torch.randint(4,(1,))) if self.augment else 0)
        layers = transformed.pop("sample_layers").reshape(length,6,3,size,size)
        return {"buffers":transformed,"layers":layers,"sequence_id":group,
                "frame_indices":list(range(start,start+length))}


def joint_inventory(root, selections=None):
    """Verify train/validation bytes and freeze metadata; never open test images."""
    root = Path(root)
    descriptor = read_v5_dataset(root)
    result = {"dataset_manifest_sha256":digest(root/"dataset.json"),"test_pixels_loaded":False,
              "partitions":descriptor["partitions"],"reference_status":descriptor["reference_status"]}
    selections = selections or {p:descriptor["partitions"][p] for p in ("train","validation")}
    if set(selections) != {"train","validation"}:
        raise ValueError("Joint inventory needs train/validation selections only")
    for p,names in selections.items():
        if (not names or len(set(names)) != len(names) or set(names)-set(descriptor["partitions"][p])):
            raise ValueError("Joint group selection crosses partition boundaries")
    result["selected"] = selections
    seen_scene,seen_images = {},{}
    for partition in ("train","validation"):
        records = []
        for group in selections[partition]:
            info = descriptor["groups"][group]
            identity = None
            for index in range(info["frame_count"]):
                _,base = verify_v5_record(root,group,index)
                scene = base.get("scene_fingerprint")
                if not isinstance(scene,str) or not scene or any(c not in "0123456789abcdef" for c in scene):
                    raise ValueError("Joint training requires valid source scene fingerprints")
                if identity is not None and scene != identity:
                    raise ValueError("Source scene settings changed inside a v5 clip")
                identity = scene
                if scene in seen_scene and seen_scene[scene] != partition:
                    raise ValueError("Scene identity crosses joint train/validation partitions")
                seen_scene[scene] = partition
                relative = Path(group)/f"frame_{index:06d}"
                layer = read_json(root/"layers"/relative/"manifest.json")
                for image_sha in (base["image_sha256"],layer["image_sha256"]):
                    if image_sha in seen_images and seen_images[image_sha] != partition:
                        raise ValueError("Image content crosses joint train/validation partitions")
                    seen_images[image_sha] = partition
                records.append({"sequence_id":group,"frame_index":index,"scene_fingerprint":scene,
                    "completion_sha256":digest(root/"frames"/relative/"manifest.json"),
                    "source_manifest_sha256":digest(root/"base"/relative/"manifest.json"),
                    "source_image_sha256":base["image_sha256"],
                    "layer_manifest_sha256":digest(root/"layers"/relative/"manifest.json"),
                    "layer_image_sha256":layer["image_sha256"]})
        result[partition] = records
    # Known reserved scene identities can be audited from JSON alone. Never
    # hash/decode their EXRs; absent test captures remain reserved.
    reserved = []
    for group in descriptor["partitions"]["test"]:
        for index in range(descriptor["groups"][group]["frame_count"]):
            path = root/"base"/group/f"frame_{index:06d}"/"manifest.json"
            if not path.exists():
                continue
            metadata = read_manifest(path)
            if metadata.get("scene_fingerprint") in seen_scene or metadata["image_sha256"] in seen_images:
                raise ValueError("Reserved test identity/content crosses joint training partitions")
            reserved.append({"sequence_id":group,"frame_index":index,"manifest_sha256":digest(path)})
    result["reserved_test_metadata"] = reserved
    return result


@contextmanager
def joint_dataset_read_lock(root):
    root = Path(root)
    read_v5_dataset(root)
    with (root/".capture.lock").open("a+") as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A capture writer is using this v5 dataset") from error
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)
