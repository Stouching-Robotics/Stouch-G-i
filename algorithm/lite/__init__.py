"""Pure-numpy runtime for the STM32 glove 21-joint live view.

Replaces torch + manotorch + polyscope with numpy + scipy + OpenCV so a fresh
machine can run ``glove_21_live_lite.py`` with a tiny dependency set.

All rotation conventions are line-by-line ports of manotorch's
``utils/geometry.py`` / ``axislayer.py`` / ``manolayer.py`` so the output
matches the calibrated MANO pipeline bit-for-bit (up to float32 rounding).
"""
