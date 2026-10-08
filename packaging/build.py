"""Build a stand-alone SEISGAMA application with PyInstaller.

    python -m pip install -e .[build]
    python packaging/build.py

Output: dist/SEISGAMA/ (Windows/Linux folder with the SEISGAMA executable)
        dist/SEISGAMA.app (macOS application bundle)
"""
import os
import sys

import PyInstaller.__main__

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

args = [
    os.path.join(HERE, "seisgama_app.py"),
    "--name=SEISGAMA",
    "--windowed",
    "--noconfirm",
    "--clean",
    f"--distpath={os.path.join(ROOT, 'dist')}",
    f"--workpath={os.path.join(ROOT, 'build')}",
    f"--specpath={os.path.join(ROOT, 'build')}",
    f"--paths={ROOT}",
    "--collect-submodules=seisgama",
    # keep the bundle small: Qt modules that SEISGAMA does not use
    *[f"--exclude-module={m}" for m in (
        "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick",
        "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
        "PySide6.QtPdf", "tkinter", "PyQt5", "PyQt6", "IPython", "pytest")],
]
if sys.platform == "darwin":
    args.append("--osx-bundle-identifier=id.ac.ugm.seisgama")

PyInstaller.__main__.run(args)
