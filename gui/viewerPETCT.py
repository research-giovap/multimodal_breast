from preprocessing import preprocessing
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QSlider, QHBoxLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

from gui.customQlabel import AspectLabel  # Custom QLabel that maintains aspect ratio

class PETCTViewer(QMainWindow):
    def __init__(self, initial_size=(512, 512)):
        super().__init__()

        # --- Volumes ---
        self.ct_volume = None
        self.pet_volume = None
        self.pet_array = None
        self.sitk_ct = None
        self.sitk_pet = None

        # --- Slice index ---
        self.current_slice = 0

        # --- Window/level for CT/PET ---
        self.ct_min = 0
        self.ct_max = 1000
        self.pet_min = 0
        self.pet_max = 12

        # --- PET opacity ---
        self.pet_opacity = 0.5

        # --- Drag state ---
        self.dragging = False
        self.last_mouse_pos = None
        self.active_volume = None  # "CT" or "PET"

        # --- PET colormap ---
        self.pet_cmap = plt.get_cmap("hot")

        # --- GUI ---
        self.label = AspectLabel() # Use the custom widget

        # Buttons
        self.btn_load_ct = QPushButton("Load CT")
        self.btn_load_pet = QPushButton("Load PET")
        self.btn_load_ct.clicked.connect(self.load_ct_dialog)
        self.btn_load_pet.clicked.connect(self.load_pet_dialog)

        # PET opacity slider
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(int(self.pet_opacity * 100))
        self.slider_opacity.valueChanged.connect(self.update_opacity)

        # Layout
        main_layout = QVBoxLayout()
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.btn_load_ct)
        buttons_layout.addWidget(self.btn_load_pet)
        buttons_layout.addWidget(QLabel("PET Opacity:"))
        buttons_layout.addWidget(self.slider_opacity)

        # Add widgets with stretch factors to prioritize the image label
        main_layout.addWidget(self.label, 1)
        main_layout.addLayout(buttons_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        self.setWindowTitle("PET/CT Viewer")
        self.resize(initial_size[0], initial_size[1])

    def select_dicom_path(self, title="Select DICOM folder"):
        folder = QFileDialog.getExistingDirectory(self, title)
        if folder:
            return folder
        return None

    def load_ct_dialog(self):
        path = self.select_dicom_path("Select CT folder")
        if path:
            self.load_ct(path)

    def load_pet_dialog(self):
        path = self.select_dicom_path("Select PET folder")
        if path:
            self.load_pet(path)

    def load_ct(self, path):
        self.sitk_ct, self.ct_volume, modality, _ = preprocessing.get_image(path)
        if modality != "CT":
            print("Warning: Loaded image is not a CT.")
        self.current_slice = self.ct_volume.shape[0] // 2
        self.ct_min = -1000
        self.ct_max = 1000
        self.update_slice()

    def load_pet(self, path):
        self.sitk_pet, self.pet_volume, modality, _ = preprocessing.get_image(path)
        if modality != "PT":
            print("Warning: Loaded image is not a PET.")
        
        if self.sitk_ct is not None:
            self.pet_resampled, self.pet_array = preprocessing.align_pet_to_ct(
                self.sitk_pet, self.sitk_ct, interpolation_method='linear'
            )
        else:
            self.pet_resampled = self.sitk_pet
            self.pet_array = self.pet_volume

        self.pet_min = float(self.pet_array.min())
        self.pet_max = float(self.pet_array.max())
        self.update_slice()

    def wheelEvent(self, event):
        if self.ct_volume is None:
            return
        if event.angleDelta().y() > 0:
            self.current_slice = min(self.current_slice + 1, self.ct_volume.shape[0] - 1)
        else:
            self.current_slice = max(self.current_slice - 1, 0)
        self.update_slice()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.label.geometry().contains(event.pos()):
                self.dragging = True
                self.last_mouse_pos = event.pos()
                if event.modifiers() & Qt.ControlModifier:
                    self.active_volume = "PET"
                    print("Adjusting PET Window/Level")
                else:
                    self.active_volume = "CT"
                    print("Adjusting CT Window/Level")

    def mouseMoveEvent(self, event):
        if self.dragging and self.last_mouse_pos is not None:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()

            if self.active_volume == "CT" and self.ct_volume is not None:
                window = self.ct_max - self.ct_min
                level = (self.ct_max + self.ct_min) / 2
                window += dx * 2
                level += -dy * 2
                self.ct_min = level - window / 2
                self.ct_max = level + window / 2
            elif self.active_volume == "PET" and self.pet_array is not None:
                window = self.pet_max - self.pet_min
                level = (self.pet_max + self.pet_min) / 2
                window += dx * 0.05
                level += -dy * 0.05
                self.pet_min = level - window / 2
                self.pet_max = level + window / 2

            self.last_mouse_pos = event.pos()
            self.update_slice()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.last_mouse_pos = None
            self.active_volume = None
            print("Window/Level adjustment finished.")

    def update_opacity(self, value):
        self.pet_opacity = value / 100.0
        self.update_slice()

    def update_slice(self):
        if self.ct_volume is None:
            self.label.set_qimage(QImage()) # Clear the image
            return

        ct_slice = self.ct_volume[self.current_slice].astype(np.float32)
        ct_norm = np.clip((ct_slice - self.ct_min) / (self.ct_max - self.ct_min + 1e-9), 0, 1)
        ct_rgb = np.stack([ct_norm] * 3, axis=-1)

        combined = ct_rgb
        if self.pet_array is not None:
            pet_slice = self.pet_array[self.current_slice].astype(np.float32)
            pet_norm = np.clip((pet_slice - self.pet_min) / (self.pet_max - self.pet_min + 1e-9), 0, 1)
            pet_rgb = self.pet_cmap(pet_norm)[..., :3]
            
            alpha = self.pet_opacity
            combined = ct_rgb * (1 - alpha) + pet_rgb * alpha
        
        combined_uint8 = (combined * 255).astype(np.uint8)
        h, w, _ = combined_uint8.shape
        qimg = QImage(combined_uint8.data, w, h, 3 * w, QImage.Format_RGB888)
        
        # Pass the final image to our custom label
        self.label.set_qimage(qimg)
        
        self.setWindowTitle(
            f"Slice {self.current_slice + 1}/{self.ct_volume.shape[0]} | "
            f"CT: [{self.ct_min:.0f}, {self.ct_max:.0f}] | "
            f"PET: [{self.pet_min:.2f}, {self.pet_max:.2f}]"
        )

