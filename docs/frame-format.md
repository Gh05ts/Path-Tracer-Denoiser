# Frame format v1

V5 retains these original paired frames unchanged under `base/` and adds a
separate [companion-layer format and reader](../dataset_generation/V5.md).
The uniform v1 count/channel contract described below remains unchanged.

One completed frame is a directory containing `manifest.json` and `frame.exr`.
The producer writes into a new sibling `.partial` directory and renames it after
both files close. Readers ignore partial directories. Existing captures are never
overwritten. This is an atomic publication convention, not an fsync durability
guarantee.

The JSON schema name is `cuda-wavefront-frame`, version `1`. Metadata includes
dimensions, sequence ID, frame and previous-frame indices, current and previous
camera bases, vertical FOV in radians, aspect ratio, render settings, absolute
sample ranges, a renderer configuration fingerprint, and the EXR's SHA-256.
Camera state is recorded explicitly; the existing renderer fingerprint alone
does not encode the camera. Sequence IDs identify one scene/trajectory capture
and must be unique within a dataset root.

New captures also include `scene_fingerprint`, a scene/config grouping key
independent of the display name, camera, and sample budget. Legacy v1 frames
remain readable, while strict whole-scene split auditing requires the new field.
This optional metadata addition does not change the EXR channels or checkpoint
formats.

EXR data and display windows are `(0,0)` through `(width-1,height-1)`, with no
subsampling. All channels are FLOAT (32-bit), except counts, which are UINT.

| Buffer | Channels | Meaning |
| --- | --- | --- |
| radiance | R, G, B | Input linear scene-referred RGB mean |
| reference | R, G, B | Independent, matched-integrator RGB mean |
| albedo | R, G, B | Texture-evaluated first-hit base color in linear RGB |
| normal | X, Y, Z | World-space shading normal, including normal maps |
| geometric_normal | X, Y, Z | World-space geometric normal |
| depth | Z | Positive distance along camera forward, in scene units |
| motion | X, Y | Previous pixel coordinate minus current pixel coordinate |
| guide_valid | Y | Binary first-hit surface mask |
| history_valid | Y | Binary previous-frustum and visibility mask |
| sample_count | N | Actual input paths per pixel |
| reference_count | N | Actual reference paths per pixel |

Channel names include their buffer prefix, for example `radiance.R`. Schema v1
contains uniform counts matching the manifest. Denoised/display images never
replace raw radiance. No exposure, tone mapping, clipping, or gamma conversion
is applied; metadata exposure is `1.0`. Miss radiance can contain sky lighting,
while its guides, depth, motion, and masks are zero.

Image origin is top left, x increases right, y increases down, and integer pixel
coordinates identify pixel centers. The separate guide pass traces at
`((x+0.5)/width, (y+0.5)/height)`. Radiance uses independent uniform subpixel
jitter. Guides are deliberately unjittered rather than an average of noisy
first-hit attributes.

For current pixel `(x,y)`, sample history at `(x+motion.X,y+motion.Y)` with
bilinear filtering. History validity requires a projected coordinate in the
previous image's pixel-center domain and a visibility ray from the previous
camera reaching the current surface. The visibility tolerance is
`max(0.001, previous_hit_distance * 0.0001)` scene units. Projection uses a small
floating-point tolerance at image edges. Invalid history can still carry a
motion vector; always use the mask. The first frame has zero motion and invalid
history. The scene is static: animated object motion and reflected/refracted
radiance motion are outside this contract.

PCG32 state is initialized from `(absolute_sample_index << 32) | pixel_index`.
For frame ordinal `i`, the capture tool chooses:

```text
stride = samples + reference_samples
input_start = sample_start + i * stride
reference_start = input_start + samples
```

The input and reference use disjoint half-open sample intervals. Different
capture invocations can reuse intervals, so choose different `sample_start`
values when requesting independent realizations of an otherwise identical
sequence. Photon maps deliberately use the existing fixed seed in both
inputs and references; increasing camera SPP does not remove photon-map bias.

The reader verifies checksum, dimensions, channel types, finite values, masks,
normal lengths, positive surface depth, sample ranges/counts, camera bases,
motion conventions, and frame/camera continuity. This validates the contract;
it does not certify reference convergence or physical estimator accuracy.

NumPy buffers use HWC layout with float32 values and uint32 counts. Tensor
conversion uses CHW layout and promotes counts to int64 for broad PyTorch
operator support, preserving their exact integer values. Sequence datasets
stack these layouts along a leading time dimension.

Capture-only lighting interventions may also record `capture_lighting` and
`geometry_fingerprint`. The latter identifies the base scene before adding
the emitter. `scene_fingerprint` and `render_fingerprint` include the canonical
lighting intervention. These fields do not change EXR channel conventions or
feature preprocessing. Lighting studies sharing geometry are within-scene
ablations and must not be described as independent scene splits.
