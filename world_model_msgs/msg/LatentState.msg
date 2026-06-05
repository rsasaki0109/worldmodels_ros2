# A model-agnostic latent representation produced by a World Model adapter.
# Stored as a flat float array plus an explicit shape so any backend
# (V-JEPA2, Cosmos, dummy, ...) can round-trip its embedding without
# committing the contract to a fixed dimensionality.

std_msgs/Header header

float32[] data        # flattened latent tensor (row-major)
uint32[] shape        # tensor shape, e.g. [1, 256] or [8, 16, 16]
string encoding       # optional tag, e.g. "vjepa2-vitl", "dummy-v0"
