"""Lightweight UI smoke test.

Does not enter the Qt event loop — just instantiates the main window,
drives the mesh-load code path on a fixture, and checks that the
bbox / unit / radius wiring produced the expected values.
"""

from pathlib import Path

import pytest

from PySide6 import QtWidgets

from fake_bevel_baker import ui

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_main_window_builds(qapp):
    w = ui.MainWindow()
    assert w.size().width() >= 800


def test_mesh_load_populates_info(qapp):
    w = ui.MainWindow()
    w._load_mesh(FIXTURES / "cube.obj")
    # Cube side length is 26.308 cm -> diag = 26.308 * sqrt(3) ~ 45.57
    assert "45." in w._info_diag.text()
    assert w._info_unit.text() == "centimeters"
    # Radius auto: 0.002 * diag
    assert abs(w._radius.value() - 0.002 * w._bbox_diag) < 1e-4
    # Radius max: 0.1 * diag
    assert abs(w._radius._max - 0.1 * w._bbox_diag) < 1e-4
    # Bake button enabled after a successful load
    assert w._bake_btn.isEnabled()


def test_uv_layout_preview_set_after_load(qapp):
    w = ui.MainWindow()
    w._load_mesh(FIXTURES / "cube.obj")
    # Preview should now have a non-null source (the UV layout).
    assert w._preview._source is not None
    img = w._preview._source
    assert img.width() > 0 and img.height() > 0


def test_detect_units_defaults_to_cm_when_missing(tmp_path):
    p = tmp_path / "no_comment.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    assert ui._detect_units(p) == "centimeters"


def test_detect_units_parses_meters(tmp_path):
    p = tmp_path / "meters.obj"
    p.write_text("# This file uses meters as units\nv 0 0 0\n")
    assert ui._detect_units(p) == "meters"
