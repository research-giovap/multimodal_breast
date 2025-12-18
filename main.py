import preprocessing.preprocessing as preprocessing
import sys
from PyQt5.QtWidgets import QApplication
from gui.viewerPETCT import PETCTViewer


def main():
    app = QApplication(sys.argv)
    viewer = PETCTViewer(initial_size=(1400, 900))
    viewer.show()
    sys.exit(app.exec_()) 

if __name__ == "__main__":
    main()