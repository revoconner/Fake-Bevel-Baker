"""
PySide6 GUI for the bevel baker.

Left panel: mesh/output pickers, angle/radius/samples sliders, resolution
and format dropdowns, dilation spin, mesh info (bbox + unit), Bake button,
progress bar. Right panel: preview - the last baked normal map, or a UV
layout on black with red edges when no bake has run yet.

Bake runs in a worker thread so the UI stays responsive. Sliders do not
trigger re-bakes; the Bake button is the only trigger.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .bake import (
    BakeResult,
    bake,
    encode_normal_to_uint16,
    reproject_to_tangent_space,
)
from .bvh import BVH
from .dilation import dilate
from .image_io import encode_normal_to_float01, write_exr_rgb32, write_png_rgb16
from .mesh_prep import PreparedMesh, prepare_mesh
from .uv_raster import rasterize_uvs


_MESH_FILTERS = (
    "Meshes (*.obj *.ply *.stl *.gltf *.glb);;"
    "OBJ (*.obj);;PLY (*.ply);;STL (*.stl);;glTF (*.gltf *.glb);;All files (*)"
)
_UNIT_KEYWORDS = ("micrometers", "millimeters", "centimeters", "meters",
                  "inches", "feet", "yards", "kilometers")


def _detect_units(path: Path) -> str:
    """Look for a unit keyword in the first 32 lines of the file.
    OBJ/PLY/STL may have a comment declaring units. Default: centimeters."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i > 32:
                    break
                low = line.lower()
                for u in _UNIT_KEYWORDS:
                    if u in low:
                        return u
    except OSError:
        pass
    return "centimeters"


def _save_tangent_png(path: Path, img_float: np.ndarray) -> None:
    write_png_rgb16(path, encode_normal_to_uint16(img_float)[::-1])


def _save_tangent_exr(path: Path, img_float: np.ndarray) -> None:
    # Same [-1, 1] -> [0, 1] encoding as PNG so viewers interpret both identically.
    write_exr_rgb32(path, encode_normal_to_float01(img_float)[::-1])


def _numpy_to_qimage_rgb8(img_float: np.ndarray) -> QtGui.QImage:
    """Convert a (H, W, 3) float [-1, 1] normal-map array to 8-bit RGB QImage."""
    v = np.clip(img_float * 0.5 + 0.5, 0.0, 1.0)
    a = (v * 255.0 + 0.5).astype(np.uint8)
    a = np.ascontiguousarray(a)
    h, w, _ = a.shape
    # Flip vertically so v=0 row sits at the image bottom visually.
    a = np.ascontiguousarray(a[::-1])
    return QtGui.QImage(a.data, w, h, w * 3, QtGui.QImage.Format_RGB888).copy()


def _uv_layout_qimage(uvs: np.ndarray, faces: np.ndarray, size: int = 1024) -> QtGui.QImage:
    """Draw UV edges as red lines on a black background."""
    img = QtGui.QImage(size, size, QtGui.QImage.Format_RGB888)
    img.fill(QtGui.QColor(0, 0, 0))
    painter = QtGui.QPainter(img)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    pen = QtGui.QPen(QtGui.QColor(220, 40, 40))
    pen.setWidthF(1.0)
    painter.setPen(pen)
    for f in faces:
        p = []
        for k in range(3):
            u, v = uvs[f[k]]
            x = float(u) * size
            y = (1.0 - float(v)) * size
            p.append(QtCore.QPointF(x, y))
        painter.drawLine(p[0], p[1])
        painter.drawLine(p[1], p[2])
        painter.drawLine(p[2], p[0])
    painter.end()
    return img


class PreviewLabel(QtWidgets.QLabel):
    """QLabel that keeps its pixmap scaled-to-fit-while-preserving-aspect."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(256, 256)
        self.setStyleSheet("background-color: #111;")
        self._source: Optional[QtGui.QImage] = None

    def set_image(self, img: Optional[QtGui.QImage]) -> None:
        self._source = img
        self._redraw()

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(e)
        self._redraw()

    def _redraw(self) -> None:
        if self._source is None:
            self.clear()
            return
        pm = QtGui.QPixmap.fromImage(self._source)
        target = self.size()
        self.setPixmap(pm.scaled(target, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


@dataclass
class BakeParams:
    mesh_path: Path
    out_path: Path
    resolution: int
    samples: int
    angle_deg: float
    radius: float
    seed: int
    dilation_px: int
    fmt: str  # "png" | "exr"
    denoise: bool = False
    denoise_quality: str = "high"  # "default" | "balanced" | "high"


class BakeWorker(QtCore.QObject):
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(object, float)  # BakeResult, elapsed
    failed = QtCore.Signal(str)

    def __init__(self, params: BakeParams):
        super().__init__()
        self.params = params

    @QtCore.Slot()
    def run(self):
        try:
            p = self.params
            t0 = time.time()
            prep = prepare_mesh(p.mesh_path, angle_deg=p.angle_deg)
            raster = rasterize_uvs(prep.tangent_uvs, prep.tangent_faces, p.resolution, p.resolution)
            bvh = BVH(prep.split_positions, prep.split_faces)
            result = bake(
                prep, raster, bvh,
                radius=p.radius, num_samples=p.samples, seed=p.seed,
                progress=lambda i, n: self.progress.emit(i, n),
            )
            tan_img = result.tangent_normal
            world_img = result.world_normal

            if p.denoise:
                from .denoise import denoise_world_normal
                ys_v, xs_v = np.where(result.valid)
                tri_ids_v = raster.tri_idx[ys_v, xs_v]
                Ng_map = np.zeros_like(world_img)
                Ng_map[ys_v, xs_v] = prep.face_normals[tri_ids_v]
                world_img = denoise_world_normal(
                    world_img, result.valid,
                    aux_normal=Ng_map,
                    quality=p.denoise_quality,
                )
                tan_img = reproject_to_tangent_space(world_img, prep, raster)

            if p.dilation_px > 0:
                tan_img, _ = dilate(tan_img, result.valid, p.dilation_px)
                world_img, _ = dilate(world_img, result.valid, p.dilation_px)

            out = p.out_path
            out.parent.mkdir(parents=True, exist_ok=True)
            if p.fmt == "png":
                _save_tangent_png(out, tan_img)
            else:
                _save_tangent_exr(out, tan_img)

            final = BakeResult(world_normal=world_img, tangent_normal=tan_img, valid=result.valid)
            self.finished.emit(final, time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class LabeledSlider(QtWidgets.QWidget):
    """Slider + spinbox pair, in a horizontal layout with a label."""
    value_changed = QtCore.Signal(float)

    def __init__(self, label: str, min_v: float, max_v: float, default: float,
                 decimals: int = 0, parent=None):
        super().__init__(parent)
        self._decimals = decimals
        self._min = float(min_v)
        self._max = float(max_v)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._label = QtWidgets.QLabel(label)
        self._label.setMinimumWidth(72)
        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setRange(0, 1000)
        if decimals > 0:
            self._spin = QtWidgets.QDoubleSpinBox()
            self._spin.setDecimals(decimals)
            self._spin.setSingleStep(10 ** (-decimals))
        else:
            self._spin = QtWidgets.QSpinBox()
            self._spin.setSingleStep(1)
        self._spin.setMinimumWidth(90)
        self._spin.setRange(min_v, max_v)

        lay.addWidget(self._label)
        lay.addWidget(self._slider, 1)
        lay.addWidget(self._spin)

        self.set_value(default)
        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

    def set_range(self, min_v: float, max_v: float) -> None:
        self._min = float(min_v)
        self._max = float(max_v)
        cur = self.value()
        self._spin.blockSignals(True)
        self._spin.setRange(min_v, max_v)
        self._spin.blockSignals(False)
        self.set_value(min(max(cur, min_v), max_v))

    def value(self) -> float:
        return float(self._spin.value())

    def set_value(self, v: float) -> None:
        v = min(max(float(v), self._min), self._max)
        self._spin.blockSignals(True)
        self._slider.blockSignals(True)
        self._spin.setValue(v)
        rng = max(self._max - self._min, 1e-9)
        self._slider.setValue(int(round((v - self._min) / rng * 1000)))
        self._spin.blockSignals(False)
        self._slider.blockSignals(False)

    def _on_slider(self, v: int) -> None:
        rng = max(self._max - self._min, 1e-9)
        val = self._min + (v / 1000.0) * rng
        self._spin.blockSignals(True)
        self._spin.setValue(val)
        self._spin.blockSignals(False)
        self.value_changed.emit(self.value())

    def _on_spin(self, v: float) -> None:
        rng = max(self._max - self._min, 1e-9)
        self._slider.blockSignals(True)
        self._slider.setValue(int(round((v - self._min) / rng * 1000)))
        self._slider.blockSignals(False)
        self.value_changed.emit(self.value())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fake Bevel Baker")
        self.resize(1280, 780)

        self._mesh_path: Optional[Path] = None
        self._prep: Optional[PreparedMesh] = None
        self._bbox_diag: float = 1.0
        self._bake_thread: Optional[QtCore.QThread] = None
        self._bake_worker: Optional[BakeWorker] = None

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        self._preview = PreviewLabel()
        splitter.addWidget(self._preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 860])
        self.setCentralWidget(splitter)

        self._status = self.statusBar()
        self._status.showMessage("Load a mesh to begin.")

    def _build_left_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(380)
        v = QtWidgets.QVBoxLayout(w)
        v.setSpacing(8)

        # --- Files ---
        files = QtWidgets.QGroupBox("Files")
        fl = QtWidgets.QFormLayout(files)
        self._mesh_edit = QtWidgets.QLineEdit()
        self._mesh_edit.setReadOnly(True)
        mb = QtWidgets.QPushButton("Browse...")
        mb.clicked.connect(self._pick_mesh)
        mh = QtWidgets.QHBoxLayout()
        mh.addWidget(self._mesh_edit, 1); mh.addWidget(mb)
        fl.addRow("Mesh:", mh)

        self._out_edit = QtWidgets.QLineEdit()
        self._out_edit.editingFinished.connect(self._on_out_edit_changed)
        ob = QtWidgets.QPushButton("Browse...")
        ob.clicked.connect(self._pick_output)
        oh = QtWidgets.QHBoxLayout()
        oh.addWidget(self._out_edit, 1); oh.addWidget(ob)
        fl.addRow("Output:", oh)
        v.addWidget(files)

        # --- Settings ---
        settings = QtWidgets.QGroupBox("Settings")
        sv = QtWidgets.QVBoxLayout(settings)

        self._angle = LabeledSlider("Angle (deg)", 5, 90, 30, decimals=0)
        self._radius = LabeledSlider("Radius", 0.0, 1.0, 0.01, decimals=4)
        self._samples = LabeledSlider("Samples", 1, 512, 16, decimals=0)
        sv.addWidget(self._angle)
        sv.addWidget(self._radius)
        sv.addWidget(self._samples)

        grid = QtWidgets.QFormLayout()
        self._res_combo = QtWidgets.QComboBox()
        for r in (512, 1024, 2048, 4096):
            self._res_combo.addItem(str(r), r)
        self._res_combo.setCurrentIndex(1)  # 1024
        grid.addRow("Resolution:", self._res_combo)

        self._fmt_combo = QtWidgets.QComboBox()
        self._fmt_combo.addItem("PNG 16-bit", "png")
        self._fmt_combo.addItem("EXR 32-bit", "exr")
        self._fmt_combo.currentIndexChanged.connect(self._on_fmt_combo_changed)
        grid.addRow("Format:", self._fmt_combo)

        self._dilation_spin = QtWidgets.QSpinBox()
        self._dilation_spin.setRange(0, 64)
        self._dilation_spin.setValue(16)
        grid.addRow("Dilation (px):", self._dilation_spin)

        self._seed_spin = QtWidgets.QSpinBox()
        self._seed_spin.setRange(0, 2**31 - 1)
        self._seed_spin.setValue(0)
        grid.addRow("Seed:", self._seed_spin)

        self._denoise_check = QtWidgets.QCheckBox("Denoise (OIDN)")
        self._denoise_check.setChecked(False)
        self._denoise_check.toggled.connect(self._on_denoise_toggled)
        grid.addRow("Post-process:", self._denoise_check)

        self._denoise_quality_combo = QtWidgets.QComboBox()
        self._denoise_quality_combo.addItem("High", "high")
        self._denoise_quality_combo.addItem("Balanced", "balanced")
        self._denoise_quality_combo.addItem("Default", "default")
        self._denoise_quality_combo.setEnabled(False)
        grid.addRow("Denoise quality:", self._denoise_quality_combo)

        sv.addLayout(grid)
        v.addWidget(settings)

        # --- Mesh info ---
        info = QtWidgets.QGroupBox("Mesh info")
        il = QtWidgets.QFormLayout(info)
        self._info_diag = QtWidgets.QLabel("-")
        self._info_dims = QtWidgets.QLabel("-")
        self._info_unit = QtWidgets.QLabel("-")
        self._info_tris = QtWidgets.QLabel("-")
        il.addRow("BBox diag:", self._info_diag)
        il.addRow("BBox X/Y/Z:", self._info_dims)
        il.addRow("Units:", self._info_unit)
        il.addRow("Triangles:", self._info_tris)
        v.addWidget(info)

        # --- Bake ---
        self._bake_btn = QtWidgets.QPushButton("Bake")
        self._bake_btn.setMinimumHeight(32)
        self._bake_btn.clicked.connect(self._on_bake)
        self._bake_btn.setEnabled(False)
        v.addWidget(self._bake_btn)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        v.addWidget(self._progress)

        v.addStretch(1)
        return w

    # ----- File pickers -----

    def _pick_mesh(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open mesh", "", _MESH_FILTERS)
        if not path:
            return
        self._load_mesh(Path(path))

    def _pick_output(self) -> None:
        fmt = self._fmt_combo.currentData()
        ext = "png" if fmt == "png" else "exr"
        default = self._mesh_path.with_suffix(f".{ext}") if self._mesh_path else Path(f"out.{ext}")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save normal map", str(default),
            "PNG (*.png);;EXR (*.exr);;All files (*)",
        )
        if path:
            self._out_edit.setText(path)
            self._last_auto_out = None  # user explicitly chose -> stop auto-replacing
            self._sync_format_from_path(path)

    # ----- Format/extension sync -----

    def _sync_format_from_path(self, path_str: str) -> None:
        """Set the format combo to match the output file extension."""
        ext = Path(path_str).suffix.lower().lstrip(".")
        target = {"exr": "exr", "png": "png"}.get(ext)
        if target is None:
            return
        for i in range(self._fmt_combo.count()):
            if self._fmt_combo.itemData(i) == target:
                if self._fmt_combo.currentIndex() != i:
                    self._fmt_combo.blockSignals(True)
                    self._fmt_combo.setCurrentIndex(i)
                    self._fmt_combo.blockSignals(False)
                return

    @QtCore.Slot()
    def _on_out_edit_changed(self) -> None:
        # User typed something -> they own the output path now.
        self._last_auto_out = None
        self._sync_format_from_path(self._out_edit.text())

    @QtCore.Slot(bool)
    def _on_denoise_toggled(self, enabled: bool) -> None:
        self._denoise_quality_combo.setEnabled(enabled)

    @QtCore.Slot(int)
    def _on_fmt_combo_changed(self, _idx: int) -> None:
        """When the user picks a format, rewrite the extension on the output path."""
        text = self._out_edit.text().strip()
        if not text:
            return
        fmt = self._fmt_combo.currentData()
        new_ext = ".png" if fmt == "png" else ".exr"
        p = Path(text)
        if p.suffix.lower() != new_ext:
            new_path = str(p.with_suffix(new_ext))
            # If the current output was an auto-derived path, keep tracking
            # the new one (with swapped extension) so a subsequent mesh
            # change still replaces it.
            was_auto = (text == getattr(self, "_last_auto_out", None))
            self._out_edit.setText(new_path)
            if was_auto:
                self._last_auto_out = new_path

    # ----- Mesh load -----

    def _load_mesh(self, path: Path) -> None:
        self._status.showMessage(f"Loading {path.name}...")
        QtWidgets.QApplication.processEvents()
        try:
            prep = prepare_mesh(path, angle_deg=float(self._angle.value()))
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Mesh load error", str(exc))
            self._status.showMessage("Mesh load failed.")
            return

        self._mesh_path = path
        self._prep = prep
        self._mesh_edit.setText(str(path))

        # Auto output path. Populate on first load, and re-derive it
        # whenever the current output path still matches the *previous*
        # mesh (i.e. the user never manually picked a specific output).
        # If the user typed their own output path, leave it alone.
        ext = "png" if self._fmt_combo.currentData() == "png" else "exr"
        new_default = str(path.with_suffix(f".{ext}"))
        current = self._out_edit.text().strip()
        is_auto = (not current) or (current == getattr(self, "_last_auto_out", None))
        if is_auto:
            self._out_edit.setText(new_default)
            self._last_auto_out = new_default

        # BBox + units
        mn = prep.tangent_positions.min(axis=0)
        mx = prep.tangent_positions.max(axis=0)
        diag = float(np.linalg.norm(mx - mn))
        unit = _detect_units(path)
        self._bbox_diag = diag
        self._info_diag.setText(f"{diag:.4f} {unit}")
        self._info_dims.setText(f"{mx[0]-mn[0]:.3f} / {mx[1]-mn[1]:.3f} / {mx[2]-mn[2]:.3f} {unit}")
        self._info_unit.setText(unit)
        self._info_tris.setText(f"{prep.tangent_faces.shape[0]} (split: {prep.split_faces.shape[0]})")

        # Radius default + range: 0 to 10% of diag, default 0.2%.
        self._radius.set_range(0.0, max(diag * 0.1, 1e-6))
        self._radius.set_value(diag * 0.002)

        # UV layout preview until first bake.
        uv_img = _uv_layout_qimage(prep.tangent_uvs, prep.tangent_faces, size=1024)
        self._preview.set_image(uv_img)

        self._bake_btn.setEnabled(True)
        self._status.showMessage(f"Loaded {path.name} ({prep.tangent_faces.shape[0]} tris)")

    # ----- Bake -----

    def _on_bake(self) -> None:
        if self._prep is None or not self._mesh_path:
            return
        out_text = self._out_edit.text().strip()
        if not out_text:
            QtWidgets.QMessageBox.warning(self, "Missing output", "Pick an output file first.")
            return
        out_path = Path(out_text)

        # Sync format combo to extension one last time (covers the case where
        # the user typed an extension and pressed Bake without tabbing out).
        self._sync_format_from_path(out_text)
        fmt = str(self._fmt_combo.currentData())

        params = BakeParams(
            mesh_path=self._mesh_path,
            out_path=out_path,
            resolution=int(self._res_combo.currentData()),
            samples=int(self._samples.value()),
            angle_deg=float(self._angle.value()),
            radius=float(self._radius.value()),
            seed=int(self._seed_spin.value()),
            dilation_px=int(self._dilation_spin.value()),
            fmt=fmt,
            denoise=bool(self._denoise_check.isChecked()),
            denoise_quality=str(self._denoise_quality_combo.currentData()),
        )

        self._set_busy(True)
        self._progress.setValue(0)
        self._status.showMessage("Baking...")

        self._bake_thread = QtCore.QThread(self)
        self._bake_worker = BakeWorker(params)
        self._bake_worker.moveToThread(self._bake_thread)
        self._bake_thread.started.connect(self._bake_worker.run)
        self._bake_worker.progress.connect(self._on_progress)
        self._bake_worker.finished.connect(self._on_bake_finished)
        self._bake_worker.failed.connect(self._on_bake_failed)
        self._bake_thread.start()

    @QtCore.Slot(int, int)
    def _on_progress(self, i: int, total: int) -> None:
        pct = int(100 * i / max(total, 1))
        self._progress.setValue(pct)

    @QtCore.Slot(object, float)
    def _on_bake_finished(self, result: BakeResult, elapsed: float) -> None:
        self._progress.setValue(100)
        self._preview.set_image(_numpy_to_qimage_rgb8(result.tangent_normal))
        self._status.showMessage(f"Bake done in {elapsed:.2f}s -> {self._out_edit.text()}")
        self._finalize_thread()
        self._set_busy(False)

    @QtCore.Slot(str)
    def _on_bake_failed(self, msg: str) -> None:
        self._finalize_thread()
        self._set_busy(False)
        self._status.showMessage("Bake failed.")
        QtWidgets.QMessageBox.critical(self, "Bake error", msg)

    def _finalize_thread(self) -> None:
        if self._bake_thread is not None:
            self._bake_thread.quit()
            self._bake_thread.wait()
            self._bake_thread = None
        self._bake_worker = None

    def _set_busy(self, busy: bool) -> None:
        self._bake_btn.setEnabled(not busy and self._prep is not None)


def run_app() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())
