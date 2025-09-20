import preprocessing as preprocessing
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication
from gui.viewerPETCT import PETCTViewer
import matplotlib.pyplot as plt
from gui.viewer import VolumeViewer
import SimpleITK as sitk

def main():
    app = QApplication(sys.argv)
    viewer = PETCTViewer(initial_size=(512, 512))
    viewer.show()
    sys.exit(app.exec_()) 

if __name__ == "__main__":
    main()