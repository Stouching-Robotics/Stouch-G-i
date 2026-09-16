# HAND+2mm Local Runtime Assets

**The model files in this directory are deliberately not tracked in git.**
This applies to both the runtime meshes and the source models:

| File | Used by |
|---|---|
| `HAND_{LEFT,RIGHT}_PINKY_PLUS_2MM.npz` | `algorithm/lite/hand.py` (`HandNumpy`) — the runtime forward layer |
| `models/MANO_{LEFT,RIGHT}.pkl` | `gui/live_3d.py` (`_load_mano_faces`) — triangle indices for the 3D view |

`.gitignore` excludes `assets/**/*.npz`, `*.pkl` and `*.bak`. The files ship with
the release package instead; model provenance, licensing, and distribution
boundaries are described in `LICENSE.txt` in this directory. Only `README.md`
and `LICENSE.txt` are tracked, on purpose — the licence notice has to travel
with the repository.

## If the models are missing

A fresh clone has neither file, and the two failure modes differ:

- **The 21-keypoint solve does not work at all.** `HandNumpy.__init__` loads
  `HAND_{side}_PINKY_PLUS_2MM.npz` directly and raises if it is absent.
- **The 3D view degrades but still starts.** `_load_mano_faces` checks
  `mano_neutral_meshes.npz` first (also untracked), then the `.pkl`, and falls
  back to `faces = None` with a warning on stderr — so the window opens without
  a hand mesh rather than crashing.

Copy both files from the release package into this directory to restore full
function.

## Asset validation

`HandNumpy` refuses an asset that is not the verified derivative, so a
substituted mesh fails loudly instead of silently producing wrong poses:

```python
if derivative_kind != "pinky_plus_2mm" or not np.isclose(
        pinky_extra_length_m, 0.002, atol=1e-12):
    raise ValueError(f"local HAND asset {filename} is not the verified +2 mm derivative")
```

`tools/adjust_hand_thickness.py` rescales the runtime mesh and writes a
matching `MANO_*_natural.pkl`, keeping that metadata intact. It stashes the
previous mesh to `*.npz.bak` (also untracked) and can undo the change with
`--restore`.

The default backend is fixed to `hand_pinky_plus_2mm` (see
`algorithm/runtime_backend.py`; the former `local_runtime_override.json`
was removed during the SDK cleanup).
