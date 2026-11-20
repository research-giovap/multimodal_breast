"""
LesionLoop Breast - interactive lesion-aware segmentation of breast cancer on [18F]FDG PET/CT.

Usage:
    python main.py
"""

import logging
import sys

from PyQt5.QtWidgets import QApplication

from gui.viewerPETCT import PETCTViewer


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    app.setApplicationName("LesionLoop Breast")
    viewer = PETCTViewer(initial_size=(1400, 900))
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
