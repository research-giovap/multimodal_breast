
# In your imports, make sure you have QSizePolicy
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QSlider, QHBoxLayout, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

class AspectLabel(QLabel):
    """
    A QLabel that maintains a fixed aspect ratio when resized.
    Corrected to handle initial display and shrinking.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pixmap = QPixmap()
        self.aspect_ratio = 1.0
        
        # --- FIX #1: Use a more forceful size policy ---
        # 'Ignored' tells the layout to completely control the widget's size,
        # overriding any minimum size hints and allowing it to shrink.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setAlignment(Qt.AlignCenter)

    def set_qimage(self, qimage: QImage):
        """
        Sets the QImage, calculates aspect ratio, and immediately updates the display.
        """
        if qimage.isNull():
            self._pixmap = QPixmap()
            self.aspect_ratio = 1.0
        else:
            self._pixmap = QPixmap.fromImage(qimage)
            # Handle potential division by zero for invalid images
            if self._pixmap.height() == 0:
                self.aspect_ratio = 1.0
            else:
                self.aspect_ratio = self._pixmap.width() / self._pixmap.height()

        # Tell the layout that our size hint may have changed due to the new aspect ratio
        self.updateGeometry()
        
        # --- FIX #2: Immediately scale and display the image ---
        # This ensures the image appears on the first load without
        # needing a manual window resize. We call the same logic that resizeEvent uses.
        self._rescale_and_set_pixmap()

    def _rescale_and_set_pixmap(self):
        """Internal helper to scale the stored pixmap to the current widget size."""
        if self._pixmap.isNull():
            super().setPixmap(QPixmap()) # Clear the pixmap
            return
            
        scaled_pixmap = self._pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        super().setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """On resize, simply call the rescale helper."""
        self._rescale_and_set_pixmap()

    def hasHeightForWidth(self):
        """Tells the layout manager we have a preferred height based on our width."""
        return not self._pixmap.isNull()

    def heightForWidth(self, width):
        """Calculates the preferred height for a given width to maintain aspect ratio."""
        if self._pixmap.isNull():
            return self.height()
        
        return int(width / self.aspect_ratio)
