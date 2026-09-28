"""Loading of network weights (strict) and of the backbone initialization used for training.

Example:
    from bevodom2.utils.checkpoints import load_weights, load_pretrained
    load_weights(model, "weights/bevodom2_nclt.pth")
    loaded = load_pretrained(model, "pretrained/bevodom2_nclt_init.pth")
"""
from collections.abc import Mapping

import torch

# Backbone sub-modules covered by the initialization file
PRETRAINED_PREFIXES = ('backbone.img_backbone.', 'backbone.img_neck.', 'backbone.depth_net.')
# Depth-bin output layer: its shape depends on d_bound, so it may be absent from the initialization file
DEPTH_BIN_LAYER = 'backbone.depth_net.depth_conv.5.'


def read_checkpoint(path, trusted_checkpoint=False):
    """Read a weight file; returns (network state_dict without `module.` prefixes, raw top-level mapping).

    Accepts a plain state_dict or {'model_state_dict': ...}. Uses weights_only unpickling unless
    trusted_checkpoint is set (only for last.pth written by the training script).
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=not trusted_checkpoint)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f'Checkpoint must be a mapping: {path}')
    state = checkpoint.get('model_state_dict', checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f'Empty or malformed network state: {path}')
    clean = {}
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f'Network state must map str to Tensor ({key!r} in {path})')
        key = key[len('module.'):] if key.startswith('module.') else key
        if key in clean:
            raise ValueError(f'Duplicate key after stripping "module.": {key}')
        clean[key] = tensor
    return clean, checkpoint


def read_state(path, trusted_checkpoint=False):
    """Return only the network state_dict."""
    return read_checkpoint(path, trusted_checkpoint)[0]


def load_weights(model, path, trusted_checkpoint=False):
    """Load network weights with strict=True; returns the raw top-level mapping."""
    state, checkpoint = read_checkpoint(path, trusted_checkpoint)
    model.load_state_dict(state, strict=True)
    return checkpoint


def pretrained_subset(model_state):
    """Model keys expected in the initialization file, and the subset allowed to be missing."""
    expected = {k for k in model_state if k.startswith(PRETRAINED_PREFIXES)}
    optional = {k for k in expected if k.startswith(DEPTH_BIN_LAYER)}
    return expected, optional


def load_pretrained(model, path):
    """Load the backbone initialization; unexpected, missing or mis-shaped keys raise. Returns the loaded keys."""
    state = read_state(path)
    model_state = model.state_dict()
    expected, optional = pretrained_subset(model_state)
    unexpected = sorted(set(state) - expected)
    missing = sorted(expected - optional - set(state))
    mismatched = sorted(k for k in set(state) & expected if state[k].shape != model_state[k].shape)
    if unexpected or missing or mismatched:
        raise RuntimeError(f'Pretrained file {path} does not match the model: '
                           f'unexpected {unexpected[:5]} ({len(unexpected)}), missing {missing[:5]} ({len(missing)}), '
                           f'shape mismatch {mismatched[:5]} ({len(mismatched)})')
    result = model.load_state_dict(state, strict=False)
    assert not result.unexpected_keys
    return set(state)
