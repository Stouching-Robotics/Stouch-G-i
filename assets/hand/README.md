# HAND+2mm Local Runtime Assets

**The model files in this directory are tracked in git and ship with the
repository.** A fresh clone has everything it needs; nothing has to be
downloaded by hand.

| File | Used by |
|---|---|
| `HAND_{LEFT,RIGHT}_PINKY_PLUS_2MM.npz` | `algorithm/lite/hand.py` (`HandNumpy`) — the runtime forward layer |
| `models/MANO_{LEFT,RIGHT}.pkl` | `gui/live_3d.py` (`_load_mano_faces`) — triangle indices for the 3D view |

The `.pkl` files are the Max-Planck MANO models; the `.npz` meshes are the
21-keypoint derivative built from them, and are what the solve and the 3D view
actually load. Provenance, licensing, and distribution boundaries are described
in `LICENSE.txt` in this directory — MANO is licensed for non-commercial
research/education use only, so the licence notice has to travel with the
repository.

`download_mano.py` in the repository root reinstalls the models from the
official source and rebuilds the derived meshes; run it with no arguments to
check the current state.

## Asset validation

`HandNumpy` refuses an asset that is not the verified derivative, so a
substituted mesh fails loudly instead of silently producing wrong poses:

```python
if derivative_kind != "pinky_plus_2mm" or not np.isclose(
        pinky_extra_length_m, 0.002, atol=1e-12):
    raise ValueError(f"local HAND asset {filename} is not the verified +2 mm derivative")
```

The default backend is fixed to `hand_pinky_plus_2mm` (see
`algorithm/runtime_backend.py`; the former `local_runtime_override.json`
was removed during the SDK cleanup).
