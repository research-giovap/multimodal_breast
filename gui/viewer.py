import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage


class VolumeViewer(QMainWindow):
    def __init__(self, volume, modality="CT", display_size=(512, 512)):
        super().__init__()
        self.volume = volume.astype(np.float32)
        self.num_slices = volume.shape[0]
        self.current_slice = 0
        self.modality = modality.upper()
        self.display_width, self.display_height = display_size

        # PET colormap
        self.pet_cmap = plt.get_cmap("hot")

        # Initial windowing
        self.win_min = float(self.volume.min())
        self.win_max = float(self.volume.max())

        # Interaction state
        self.dragging = False
        self.last_mouse_pos = None

        # QLabel for image
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(self.display_width, self.display_height)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Set window title
        self.setWindowTitle("Volume Viewer")

        # Display first slice
        self.update_slice(self.current_slice)

    def apply_windowing(self, img):
        norm = np.clip((img - self.win_min) / (self.win_max - self.win_min), 0, 1)
        if self.modality == "PT":  # PET with hot colormap
            colored = (self.pet_cmap(norm)[..., :3] * 255).astype(np.uint8)
            h, w, _ = colored.shape
            qimg = QImage(colored.data, w, h, 3 * w, QImage.Format_RGB888)
        else:  # CT grayscale
            gray = (norm * 255).astype(np.uint8)
            h, w = gray.shape
            qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
        return qimg

    def update_slice(self, idx):
        self.current_slice = np.clip(idx, 0, self.num_slices - 1)
        slice_img = self.volume[self.current_slice]
        qimg = self.apply_windowing(slice_img)
        pixmap = QPixmap.fromImage(qimg)

        # Scale the pixmap to fixed display size
        pixmap = pixmap.scaled(
            self.display_width,
            self.display_height,
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )

        self.label.setPixmap(pixmap)
        self.setWindowTitle(
            f"Slice {self.current_slice + 1}/{self.num_slices} "
            f"[Window: {self.win_min:.2f}-{self.win_max:.2f}]"
        )

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.update_slice(self.current_slice + 1)
        else:
            self.update_slice(self.current_slice - 1)

    # --- Window/level interaction ---
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging and self.last_mouse_pos is not None:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()

            # Window (contrast) change with horizontal drag
            window = self.win_max - self.win_min
            window += dx * 0.01 * window

            # Level (center) change with vertical drag
            level = (self.win_max + self.win_min) / 2
            level += -dy * 0.01 * window  # invert y-axis for natural behavior

            # Update min/max
            window = max(window, 1e-3)
            self.win_min = level - window / 2
            self.win_max = level + window / 2

            self.last_mouse_pos = event.pos()
            self.update_slice(self.current_slice)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.last_mouse_pos = None


def view_volume(volume, modality="CT", display_size=(512, 512)):
    app = QApplication(sys.argv)
    viewer = VolumeViewer(volume, modality=modality, display_size=display_size)
    viewer.show()
    sys.exit(app.exec_())