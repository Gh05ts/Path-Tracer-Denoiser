"""The explicitly audited BMW integrator-11 source / integrator-12 layer bridge.

This exception is limited to the two binaries and BMW comparison preserved in
the v4 capture history. It does not declare arbitrary integrators equivalent.
"""

BMW11_RENDERER = "e4c694544e3491fb0a3bdb2923b461065817759f0ed3c1f891460a164bba6a52"
BMW12_RENDERER = "ad8a93255f5816f14864e69536fb914a541c6a3eb94ac054c0b32626588d2fa2"
BMW_VALIDATION = "ee506a00532bb6c2ac9967ed88417bf7df2f7cf949afc40d34678c2f6896b10c"


def integrator12_fingerprint(value):
    """Replace only the final little-endian uint32 in the renderer's FNV-1a key."""
    if not isinstance(value, str) or not 1 <= len(value) <= 16 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid source render fingerprint")
    mask, prime = (1 << 64) - 1, 1099511628211
    inverse = pow(prime, -1, 1 << 64)
    result = int(value, 16)
    for byte in reversed((11, 0, 0, 0)):
        result = ((result * inverse) & mask) ^ byte
    for byte in (12, 0, 0, 0):
        result = ((result ^ byte) * prime) & mask
    return format(result, "x")


def bmw_source_binding(base):
    if base.get("scene") != "bmw" or base.get("render", {}).get("integrator_version") != 11:
        raise ValueError("The audited integrator bridge applies only to BMW integrator-11 sources")
    return {"format": "wavefront-bmw-integrator12-backfill", "version": 1,
            "source_image_sha256": base["image_sha256"],
            "source_render_fingerprint": base["render_fingerprint"],
            "layer_renderer_sha256": BMW12_RENDERER, "validation_sha256": BMW_VALIDATION}


def compatible_bmw_layer(metadata, base):
    if "source_compatibility" not in metadata:
        return False
    try:
        return (metadata["source_compatibility"] == bmw_source_binding(base)
                and metadata["render"].get("integrator_version") == 12
                and metadata["render_fingerprint"] == integrator12_fingerprint(base["render_fingerprint"]))
    except (KeyError, ValueError, TypeError):
        return False
