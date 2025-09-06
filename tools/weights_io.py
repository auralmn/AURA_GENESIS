import os
import json
import numpy as np
from typing import Any, Dict


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def save_network_weights(net: Any, base_dir: str) -> Dict[str, int]:
    """Best-effort saver for common network components.
    Saves thalamus, hippocampus, router groups, and any '*.era_specialists'.
    Returns counts per component.
    """
    _ensure_dir(base_dir)
    counts: Dict[str, int] = {}

    # Thalamus
    th = getattr(net, '_thalamus', None)
    if th is not None:
        try:
            W = np.vstack([n.nlms_head.w for n in th.neurons])
            np.savez_compressed(os.path.join(base_dir, 'thalamus_weights.npz'), W=W)
            counts['thalamus'] = W.shape[0]
        except Exception:
            counts['thalamus'] = 0

    # Hippocampus
    hip = getattr(net, '_hippocampus', None)
    if hip is not None:
        try:
            W = np.vstack([n.nlms_head.w for n in hip.neurons])
            np.savez_compressed(os.path.join(base_dir, 'hippocampus_weights.npz'), W=W)
            counts['hippocampus'] = W.shape[0]
        except Exception:
            counts['hippocampus'] = 0

    # Thalamic router groups
    router = getattr(net, '_thalamic_router', None)
    if router is not None:
        saved = 0
        try:
            for group_name, group in getattr(router, 'routing_neurons', {}).items():
                try:
                    W = np.vstack([n.nlms_head.w for n in group])
                    np.savez_compressed(os.path.join(base_dir, f'router_{group_name}.npz'), W=W)
                    saved += W.shape[0]
                except Exception:
                    pass
        except Exception:
            pass
        counts['router'] = saved

    # Era specialists in historical network
    era_specs = getattr(net, 'era_specialists', None)
    if era_specs is not None and isinstance(era_specs, dict):
        try:
            payload = { name: np.asarray(n.nlms_head.w) for name, n in era_specs.items() }
            np.savez_compressed(os.path.join(base_dir, 'era_specialists.npz'), **payload)
            counts['era_specialists'] = len(payload)
        except Exception:
            counts['era_specialists'] = 0

    return counts


def load_network_weights(net: Any, base_dir: str) -> Dict[str, int]:
    """Best-effort loader to restore weights saved by save_network_weights.
    Returns counts of items loaded per component.
    """
    counts: Dict[str, int] = {}
    if not os.path.isdir(base_dir):
        return counts

    # Thalamus
    th = getattr(net, '_thalamus', None)
    th_path = os.path.join(base_dir, 'thalamus_weights.npz')
    if th is not None and os.path.isfile(th_path):
        try:
            data = np.load(th_path)
            W = data['W']
            loaded = 0
            for i, n in enumerate(getattr(th, 'neurons', []) or []):
                if i < W.shape[0] and W.shape[1] == n.nlms_head.w.shape[0]:
                    n.nlms_head.w = W[i].astype(np.float64)
                    loaded += 1
            counts['thalamus'] = loaded
        except Exception:
            counts['thalamus'] = 0

    # Hippocampus
    hip = getattr(net, '_hippocampus', None)
    hip_path = os.path.join(base_dir, 'hippocampus_weights.npz')
    if hip is not None and os.path.isfile(hip_path):
        try:
            data = np.load(hip_path)
            W = data['W']
            loaded = 0
            for i, n in enumerate(getattr(hip, 'neurons', []) or []):
                if i < W.shape[0] and W.shape[1] == n.nlms_head.w.shape[0]:
                    n.nlms_head.w = W[i].astype(np.float64)
                    loaded += 1
            counts['hippocampus'] = loaded
        except Exception:
            counts['hippocampus'] = 0

    # Router groups
    router = getattr(net, '_thalamic_router', None)
    if router is not None:
        loaded_total = 0
        try:
            for group_name, group in getattr(router, 'routing_neurons', {}).items():
                path = os.path.join(base_dir, f'router_{group_name}.npz')
                if not os.path.isfile(path):
                    continue
                try:
                    data = np.load(path)
                    W = data['W']
                    for i, n in enumerate(group):
                        if i < W.shape[0] and W.shape[1] == n.nlms_head.w.shape[0]:
                            n.nlms_head.w = W[i].astype(np.float64)
                            loaded_total += 1
                except Exception:
                    pass
        except Exception:
            pass
        counts['router'] = loaded_total

    # Era specialists
    era_specs = getattr(net, 'era_specialists', None)
    era_path = os.path.join(base_dir, 'era_specialists.npz')
    if era_specs is not None and isinstance(era_specs, dict) and os.path.isfile(era_path):
        try:
            data = np.load(era_path)
            loaded = 0
            for name, n in era_specs.items():
                if name in data and data[name].shape == n.nlms_head.w.shape:
                    n.nlms_head.w = data[name].astype(np.float64)
                    loaded += 1
            counts['era_specialists'] = loaded
        except Exception:
            counts['era_specialists'] = 0

    return counts
