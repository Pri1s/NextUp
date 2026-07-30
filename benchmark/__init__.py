"""Independent benchmark for court calibration.

This package measures the calibration engine; it is never part of it. The
separation is structural, not stylistic: an annotation that has seen the model's
predictions inherits the model's systematic error, and a benchmark built from such
annotations would certify the very drift it exists to detect (design plan §7.2).

So ``benchmark`` may import ``contracts``, and the model-free geometry in
``calibration.estimator``, ``calibration.io.video`` and ``calibration.viz``. It
must never import ``calibration.detectors``, ``calibration.adapters`` or
``calibration.calibrator``. ``tests/test_blindness.py`` enforces that.
"""
