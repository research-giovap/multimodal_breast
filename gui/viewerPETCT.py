"""
LesionLoop Breast - main window.

Loads a CT and a PET DICOM series, lets the operator draw a click or scribble on a breast lesion,
runs the 2D or 3D segmentation and shows the metabolic tumour volume, SUVmax, SUVmean and TLG.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch

from preprocessing import preprocessing
from segmentation.inference import (load_method, available_models, ensemble_available, segment_slice,
                                    segment_volume, VALIDATED_ENSEMBLE_LABEL)

from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QWidget, QApplication,
    QPushButton, QFileDialog, QSlider, QHBoxLayout,
    QSpinBox, QRadioButton, QButtonGroup, QGroupBox,
    QComboBox, QFrame, QMessageBox, QCheckBox, QProgressDialog,
    QGridLayout, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPalette, QColor

try:
    from gui.customQlabel import AspectLabel
except ImportError:
    from customQlabel import AspectLabel

class PETCTViewer(QMainWindow):
    def __init__(self, initial_size=(800, 800)):
        super().__init__()

        # --- Volumes (Visualization) ---
        self.ct_volume = None
        self.pet_volume = None
        self.pet_array = None
        self.sitk_ct = None
        self.sitk_pet = None
        self.pet_resampled = None

        # --- Volumes (AI / Segmentation Optimized) ---
        self.ai_ct_norm = None
        self.ai_pet_norm = None
        self.ai_scale_info = None

        # --- Deep Learning Model ---
        self.model = None
        self._model_cache = {}          # loaded models, by combo entry
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "segmentation", "models")
        
        # --- Masks ---
        self.seed_mask = None           # User Input (Red)
        self.segmentation_mask = None   # Single Slice Result (Cyan)
        self.volumetric_mask = None     # Propagated 3D Result (Purple)

        # --- Brush Colors ---
        self.brush_colors = {
            1: ("Red",    (1.0, 0.0, 0.0)),
            2: ("Green",  (0.0, 1.0, 0.0)),
            3: ("Blue",   (0.0, 0.5, 1.0)),
            4: ("Yellow", (1.0, 1.0, 0.0)),
        }
        self.current_color_id = 1
        self.current_slice = 0

        # --- Visualization Settings ---
        self.ct_min, self.ct_max = -1000, 1000
        self.pet_min, self.pet_max = 0, 12
        self.pet_opacity = 0.5
        self.seg_opacity = 0.5 
        self.pet_cmap = plt.get_cmap("hot")
        
        # Layer Visibility
        self.show_ct = True
        self.show_pet = True
        self.show_seeds = True
        self.show_seg = True         # Show Single Slice Result
        self.show_volumetric = True  # Show 3D Result

        self.dragging = False
        self.last_mouse_pos = None
        self.active_action = None 

        self.set_theme()
        self.init_ui()
        self.resize(initial_size[0], initial_size[1])

        self.refresh_model_list()
        QTimer.singleShot(100, self.force_resize_event)

    def set_theme(self):
        p = self.palette()
        p.setColor(QPalette.Window, QColor(40, 40, 40))
        p.setColor(QPalette.WindowText, Qt.white)
        self.setPalette(p)

    def init_ui(self):
        self.setWindowTitle("LesionLoop Breast")

        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ==========================================
        # LEFT COLUMN
        # ==========================================
        self.left_panel = QFrame()
        self.left_panel.setFixedWidth(320)
        self.left_panel.setStyleSheet("""
            QFrame { background-color: #333333; border-right: 1px solid #444; }
            QLabel { color: #ddd; font-weight: bold; font-size: 13px; margin-top: 5px; }
            QPushButton { 
                background-color: #555; color: white; border: 1px solid #666; 
                padding: 8px; border-radius: 4px; font-size: 12px; margin-bottom: 5px;
            }
            QPushButton:hover { background-color: #666; }
            QGroupBox { color: #eee; font-weight: bold; border: 1px solid #555; border-radius: 5px; margin-top: 10px; padding-top: 15px; }
            QComboBox { background-color: #555; color: white; padding: 5px; border: 1px solid #666; }
            QSpinBox { background-color: #555; color: white; padding: 5px; border: 1px solid #666; }
            QCheckBox { color: #ddd; spacing: 5px; }
        """)
        
        # Scrollable content, so that the column fits also on small screens
        left_outer = QVBoxLayout(self.left_panel)
        left_outer.setContentsMargins(0, 0, 0, 0)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet("""
            QScrollArea { border: none; background-color: #333333; }
            QScrollBar:vertical { background: #333333; width: 8px; margin: 0; }
            QScrollBar::handle:vertical { background: #555; border-radius: 4px; min-height: 30px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)
        left_content = QWidget()
        left_content.setObjectName("leftContent")
        left_content.setStyleSheet("QWidget#leftContent { background-color: #333333; }")
        left_content.setFixedWidth(self.left_panel.width() - 9)   # same width as before; room for the scroll bar
        left_scroll.setWidget(left_content)
        left_outer.addWidget(left_scroll)

        menu_layout = QVBoxLayout(left_content)
        menu_layout.setContentsMargins(15, 20, 15, 20)
        menu_layout.setSpacing(15)

        # --- 1. DATA ---
        lbl_data = QLabel("DATA LOADING")
        self.btn_load_ct = QPushButton("Load CT Folder")
        self.btn_load_pet = QPushButton("Load PET Folder")
        self.btn_load_ct.clicked.connect(self.load_ct_dialog)
        self.btn_load_pet.clicked.connect(self.load_pet_dialog)
        menu_layout.addWidget(lbl_data)
        menu_layout.addWidget(self.btn_load_ct)
        menu_layout.addWidget(self.btn_load_pet)

        # --- 2. LAYERS ---
        self.grp_layers = QGroupBox("VISIBLE LAYERS")
        layer_layout = QVBoxLayout()
        
        self.chk_ct = QCheckBox("Show CT")
        self.chk_ct.setChecked(True)
        self.chk_ct.toggled.connect(lambda v: setattr(self, 'show_ct', v) or self.update_slice())
        
        h_pet = QHBoxLayout()
        self.chk_pet = QCheckBox("Show PET")
        self.chk_pet.setChecked(True)
        self.chk_pet.toggled.connect(lambda v: setattr(self, 'show_pet', v) or self.update_slice())
        self.slider_op = QSlider(Qt.Horizontal)
        self.slider_op.setRange(0, 100); self.slider_op.setValue(50)
        self.slider_op.valueChanged.connect(self.update_opacity)
        h_pet.addWidget(self.chk_pet); h_pet.addWidget(self.slider_op)

        self.chk_seeds = QCheckBox("Show Seeds (Input)")
        self.chk_seeds.setStyleSheet("color: #ff6666;") 
        self.chk_seeds.setChecked(True)
        self.chk_seeds.toggled.connect(lambda v: setattr(self, 'show_seeds', v) or self.update_slice())

        # Single Slice Result (Cyan)
        h_seg = QHBoxLayout()
        self.chk_seg = QCheckBox("Show 2D Result")
        self.chk_seg.setStyleSheet("color: #66ff66;") 
        self.chk_seg.setChecked(True)
        self.chk_seg.toggled.connect(lambda v: setattr(self, 'show_seg', v) or self.update_slice())
        self.slider_seg = QSlider(Qt.Horizontal)
        self.slider_seg.setRange(0, 100); self.slider_seg.setValue(50)
        self.slider_seg.valueChanged.connect(self.update_seg_opacity)
        h_seg.addWidget(self.chk_seg); h_seg.addWidget(self.slider_seg)

        # Volumetric Result (Purple)
        self.chk_vol = QCheckBox("Show Volumetric")
        self.chk_vol.setStyleSheet("color: #d880f0;") 
        self.chk_vol.setChecked(True)
        self.chk_vol.toggled.connect(lambda v: setattr(self, 'show_volumetric', v) or self.update_slice())

        layer_layout.addWidget(self.chk_ct)
        layer_layout.addLayout(h_pet)
        layer_layout.addWidget(self.chk_seeds)
        layer_layout.addLayout(h_seg)
        layer_layout.addWidget(self.chk_vol)
        
        self.btn_reset = QPushButton("Reset View")
        self.btn_reset.clicked.connect(self.reset_view)
        layer_layout.addWidget(self.btn_reset)

        self.grp_layers.setLayout(layer_layout)
        menu_layout.addWidget(self.grp_layers)

        # --- 3. SEGMENTATION ---
        self.grp_seg = QGroupBox("SEGMENTATION")
        self.grp_seg.setStyleSheet("QGroupBox { background-color: #3a3a3a; border: 1px solid #0078d7; }")
        seg_layout = QVBoxLayout()
        
        # Tools
        row_tools = QHBoxLayout()
        self.grp_tools = QButtonGroup(self)
        self.rad_nav = QRadioButton("Nav"); self.rad_draw = QRadioButton("Draw"); self.rad_erase = QRadioButton("Erase")
        self.rad_nav.setChecked(True)
        self.grp_tools.addButton(self.rad_nav); self.grp_tools.addButton(self.rad_draw); self.grp_tools.addButton(self.rad_erase)
        row_tools.addWidget(self.rad_nav); row_tools.addWidget(self.rad_draw); row_tools.addWidget(self.rad_erase)
        seg_layout.addLayout(row_tools)

        # Brush
        row_brush = QHBoxLayout()
        self.spin_size = QSpinBox(); self.spin_size.setRange(1, 50); self.spin_size.setValue(3)
        self.combo_col = QComboBox()
        for i, (n, _) in self.brush_colors.items(): self.combo_col.addItem(n, i)
        self.combo_col.currentIndexChanged.connect(lambda: setattr(self, 'current_color_id', self.combo_col.currentData()))
        row_brush.addWidget(self.spin_size); row_brush.addWidget(self.combo_col)
        seg_layout.addLayout(row_brush)

        # Model
        self.combo_model = QComboBox()
        seg_layout.addWidget(self.combo_model)

        # --- SEPARATE BUTTONS ---
        self.btn_seg_2d = QPushButton("Segment Slice (2D)")
        self.btn_seg_2d.setStyleSheet("background-color: #0078d7; font-weight: bold; padding: 10px; margin-top: 5px;")
        self.btn_seg_2d.clicked.connect(self.run_segmentation_2d)

        self.btn_seg_3d = QPushButton("Segment Volumetric (3D)")
        self.btn_seg_3d.setStyleSheet("background-color: #8e44ad; font-weight: bold; padding: 10px; margin-top: 5px;")
        self.btn_seg_3d.clicked.connect(self.run_segmentation_3d)

        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.setStyleSheet("background-color: #c0392b;")
        self.btn_clear.clicked.connect(self.clear_mask)

        self.btn_save = QPushButton("Save 3D Mask (NIfTI)")
        self.btn_save.setStyleSheet("background-color: #27ae60; padding: 6px; margin-top: 5px;")
        self.btn_save.clicked.connect(self.save_volumetric_mask)

        seg_layout.addWidget(self.btn_seg_2d)
        seg_layout.addWidget(self.btn_seg_3d)
        seg_layout.addWidget(self.btn_save)
        seg_layout.addWidget(self.btn_clear)
        self.grp_seg.setLayout(seg_layout)
        menu_layout.addWidget(self.grp_seg)

        # --- 4. QUANTITATIVE SUMMARY (volumetric segmentation) ---
        self.grp_summary = QGroupBox("QUANTITATIVE SUMMARY")
        self.grp_summary.setStyleSheet("QGroupBox { background-color: #3a3a3a; border: 1px solid #8e44ad; }")
        summary_layout = QGridLayout()
        summary_layout.setHorizontalSpacing(10)
        summary_layout.setVerticalSpacing(6)
        name_style = "color: #bbbbbb; font-weight: normal; font-size: 12px; margin-top: 0px; border: none; background: transparent;"
        value_style = "color: white; font-weight: bold; font-size: 13px; margin-top: 0px; border: none; background: transparent;"
        self.summary_values = {}
        rows = [("mtv", "MTV"), ("slices", "Slices"), ("suvmax", "SUVmax"), ("suvmean", "SUVmean"), ("tlg", "TLG")]
        for r, (key, name) in enumerate(rows):
            lbl_name = QLabel(name)
            lbl_name.setStyleSheet(name_style)
            lbl_val = QLabel("—")
            lbl_val.setStyleSheet(value_style)
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl_val.setTextInteractionFlags(Qt.TextSelectableByMouse)   # values can be copied
            summary_layout.addWidget(lbl_name, r, 0)
            summary_layout.addWidget(lbl_val, r, 1)
            self.summary_values[key] = lbl_val
        self.lbl_summary_note = QLabel("Run a volumetric (3D) segmentation")
        self.lbl_summary_note.setStyleSheet("color: #888888; font-weight: normal; font-style: italic; "
                                            "font-size: 11px; margin-top: 4px; border: none; background: transparent;")
        self.lbl_summary_note.setWordWrap(True)
        summary_layout.addWidget(self.lbl_summary_note, len(rows), 0, 1, 2)
        self.grp_summary.setLayout(summary_layout)
        menu_layout.addWidget(self.grp_summary)
        menu_layout.addStretch()

        # ==========================================
        # RIGHT COLUMN (Viewer)
        # ==========================================
        self.right_panel = QWidget()
        self.right_panel.setStyleSheet("background-color: black;")
        
        # 1. The Viewer Layout
        self.viewer_layout = QVBoxLayout(self.right_panel)
        self.viewer_layout.setContentsMargins(0,0,0,0) # No margins so image hits edge
        
        # 2. Add the Image Label
        self.label = AspectLabel()
        self.viewer_layout.addWidget(self.label)

        # 3. Add the Toggle Button (Floating Overlay)
        # We set parent to right_panel so it sits ON TOP of the viewer
        self.btn_toggle_menu = QPushButton("☰", self.right_panel) 
        self.btn_toggle_menu.setFixedSize(40, 40)
        self.btn_toggle_menu.move(10, 10) # Position 10px from top-left corner
        self.btn_toggle_menu.setCursor(Qt.PointingHandCursor)
        
        # Style it to look like a floating tool
        self.btn_toggle_menu.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 60, 0.8); 
                color: white; 
                font-size: 20px; 
                font-weight: bold;
                border: 1px solid #555; 
                border-radius: 5px;
            }
            QPushButton:hover { 
                background-color: rgba(0, 120, 215, 1.0); 
                border-color: #0078d7;
            }
        """)
        self.btn_toggle_menu.clicked.connect(self.toggle_menu)
        
        # IMPORTANT: raise_() ensures it paints ON TOP of the medical image
        self.btn_toggle_menu.raise_()

        main_layout.addWidget(self.left_panel)
        main_layout.addWidget(self.right_panel, 1)
        self.setCentralWidget(central_widget)

    def force_resize_event(self):
        s = self.size(); self.resize(s.width()+1, s.height()+1); self.resize(s.width(), s.height())

    def create_progress(self, title, message):
        progress = QProgressDialog(message, None, 0, 0, self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        return progress

    # --- MODEL & DATA ---
    def refresh_model_list(self):
        """First entry: the validated ensemble (default). Then the single models, for comparison."""
        self.combo_model.clear()
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder, exist_ok=True)
        if ensemble_available(self.model_folder):
            self.combo_model.addItem(VALIDATED_ENSEMBLE_LABEL, "ensemble")
        for file_name, label in available_models(self.model_folder):
            self.combo_model.addItem(f"{label} ({file_name})", file_name)
        if self.combo_model.count():
            self.combo_model.setEnabled(True)
            self.combo_model.setCurrentIndex(0)
        else:
            self.combo_model.addItem("No models found", None)
            self.combo_model.setEnabled(False)

    def load_selected_model(self):
        key = self.combo_model.currentData()
        if not key:
            return False
        if key in self._model_cache:
            self.model = self._model_cache[key]
            return True
        try:
            progress = self.create_progress("Loading Model", f"Loading {self.combo_model.currentText()}...")
            self.model = load_method(self.model_folder, key, self.device)
            self._model_cache[key] = self.model
            progress.close()
            return True
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
            return False

    # --- BUTTON 1: SINGLE SLICE ---
    def run_segmentation_2d(self):
        if self.ai_ct_norm is None: QMessageBox.warning(self, "Error", "Load CT and PET first."); return
        if self.seed_mask is None or not np.any(self.seed_mask[self.current_slice]):
            QMessageBox.warning(self, "Error", "Draw a prompt on the current slice."); return
        if not self.load_selected_model(): return
        try:
            pred, meta = segment_slice(self.model, self.device, self.ai_ct_norm, self.ai_pet_norm,
                                       self.ai_scale_info, self.seed_mask[self.current_slice], self.current_slice)
            self.segmentation_mask = preprocessing.reconstruct_and_resample_back_optimized(
                pred, meta, self.sitk_ct).astype(np.uint8)
            self.update_slice()
            if not pred.any():
                QMessageBox.information(self, "Result", "No lesion segmented on this slice.\n"
                                        "The prompt may lie on non-tumoural uptake.")
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Error", f"{e}")

    # --- BUTTON 2: VOLUMETRIC ---
    def run_segmentation_3d(self):
        if self.ai_ct_norm is None: QMessageBox.warning(self, "Error", "Load CT and PET first."); return
        if self.seed_mask is None or not np.any(self.seed_mask[self.current_slice]):
            QMessageBox.warning(self, "Error", "Draw a prompt on the current slice."); return
        if not self.load_selected_model(): return
        progress = None
        try:
            progress = self.create_progress("Volumetric Segmentation", "Starting...")

            def report(text):
                progress.setLabelText(text)
                QApplication.processEvents()

            ai_mask, info = segment_volume(self.model, self.device, self.ai_ct_norm, self.ai_pet_norm,
                                           self.ai_scale_info, self.seed_mask[self.current_slice],
                                           self.current_slice, progress=report)
            if not ai_mask.any():
                progress.close()
                if self.volumetric_mask is not None:
                    self.volumetric_mask[:] = 0
                    self.update_slice()
                self.update_summary_panel()
                QMessageBox.information(self, "Result", "No lesion segmented.\n"
                                        "The prompt may lie on non-tumoural uptake.")
                return
            report("Resampling to the CT grid...")
            self.volumetric_mask = preprocessing.resample_full_mask_back_to_viewer(
                ai_mask.astype(np.uint8), self.ai_scale_info['sitk_resampled_ref'], self.sitk_ct).astype(np.uint8)
            progress.close()
            self.update_slice()
            self.update_summary_panel()
        except Exception as e:
            if progress is not None: progress.close()
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Error", f"{e}")

    def mask_stats(self):
        """Volume and SUV statistics of the volumetric mask (CT grid). None if there is no mask."""
        if self.volumetric_mask is None or self.sitk_ct is None or not np.any(self.volumetric_mask):
            return None
        m = self.volumetric_mask > 0
        vox_ml = float(np.prod(self.sitk_ct.GetSpacing())) / 1000.0
        vol = float(m.sum() * vox_ml)
        st = {"mtv": vol, "slices": int(m.any(axis=(1, 2)).sum())}
        if self.pet_array is not None:
            v = self.pet_array[m]
            st.update({"suvmax": float(v.max()), "suvmean": float(v.mean()), "tlg": float(v.mean() * vol)})
        return st

    def mask_summary(self):
        """Text version of mask_stats (used in the save dialog)."""
        st = self.mask_stats()
        if st is None:
            return "No volumetric segmentation."
        lines = [f"Metabolic tumour volume: {st['mtv']:.2f} mL", f"Slices: {st['slices']}"]
        if "suvmax" in st:
            lines += [f"SUVmax: {st['suvmax']:.2f}", f"SUVmean: {st['suvmean']:.2f}", f"TLG: {st['tlg']:.2f} g"]
        return "\n".join(lines)

    def update_summary_panel(self):
        """Shows the statistics of the current volumetric mask in the QUANTITATIVE SUMMARY box."""
        if not hasattr(self, "summary_values"):
            return
        st = self.mask_stats()
        fmt = {"mtv": lambda x: f"{x:.2f} mL", "slices": lambda x: f"{x}", "suvmax": lambda x: f"{x:.2f}",
               "suvmean": lambda x: f"{x:.2f}", "tlg": lambda x: f"{x:.2f} g"}
        for key, lbl in self.summary_values.items():
            lbl.setText(fmt[key](st[key]) if (st is not None and key in st) else "—")
        if st is None:
            self.lbl_summary_note.setText("Run a volumetric (3D) segmentation")
        else:
            self.lbl_summary_note.setText(f"Model: {self.combo_model.currentText()}")

    def save_volumetric_mask(self):
        if self.volumetric_mask is None or not np.any(self.volumetric_mask):
            QMessageBox.warning(self, "Error", "No volumetric segmentation to save."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save mask", "lesion_mask.nii.gz", "NIfTI (*.nii.gz *.nii)")
        if not path:
            return
        if not path.endswith((".nii", ".nii.gz")):
            path += ".nii.gz"
        img = sitk.GetImageFromArray(self.volumetric_mask.astype(np.uint8))
        img.CopyInformation(self.sitk_ct)            # same geometry as the loaded CT
        sitk.WriteImage(img, path)
        QMessageBox.information(self, "Saved", f"Mask saved to:\n{path}\n\n" + self.mask_summary())

    # --- LOADING & UI ---
    def toggle_menu(self):
        if self.left_panel.isVisible():
            self.left_panel.hide()
            self.btn_toggle_menu.setText("➤") # Arrow indicating "Expand"
        else:
            self.left_panel.show()
            self.btn_toggle_menu.setText("☰") # Hamburger indicating "Menu"

    def load_ct_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Select CT")
        if path:
            p = self.create_progress("Load CT", "Reading..."); 
            self.sitk_ct, self.ct_volume, _, _ = preprocessing.get_image(path)
            self.seed_mask = np.zeros_like(self.ct_volume, dtype=np.uint8)
            self.segmentation_mask = np.zeros_like(self.ct_volume, dtype=np.uint8)
            self.volumetric_mask = np.zeros_like(self.ct_volume, dtype=np.uint8)
            self.current_slice = self.ct_volume.shape[0]//2
            self.ai_ct_norm = None; p.close(); self.update_slice()
            self.update_summary_panel()

    def load_pet_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Select PET")
        if path and self.sitk_ct:
            p = self.create_progress("Load PET", "Reading...")
            try:
                self.sitk_pet, self.pet_volume, _, _ = preprocessing.get_image(path)
                p.setLabelText("Aligning..."); QApplication.processEvents()
                self.pet_resampled, self.pet_array = preprocessing.align_pet_to_ct(self.sitk_pet, self.sitk_ct)
                self.pet_min, self.pet_max = float(self.pet_array.min()), float(self.pet_array.max())
                
                p.setLabelText("Optimizing for AI..."); QApplication.processEvents()
                self.ai_ct_norm, self.ai_pet_norm, self.ai_scale_info = preprocessing.prepare_volumes_for_inference(
                    self.sitk_ct, self.ct_volume, self.pet_resampled, self.pet_array
                )
            except Exception as e: QMessageBox.critical(self, "Error", f"{e}")
            p.close(); self.update_slice()

    def reset_view(self): self.label.reset_view()
    def clear_mask(self): 
        if self.seed_mask is not None: self.seed_mask[:] = 0
        if self.segmentation_mask is not None: self.segmentation_mask[:] = 0
        if self.volumetric_mask is not None: self.volumetric_mask[:] = 0
        self.update_slice()
        self.update_summary_panel()
    
    def update_opacity(self, v): self.pet_opacity = v/100; self.update_slice()
    def update_seg_opacity(self, v): self.seg_opacity = v/100; self.update_slice()

    def update_slice(self):
        if self.ct_volume is None: self.label.set_qimage(QImage()); return
        
        # 1. CT
        if self.show_ct:
            ct = self.ct_volume[self.current_slice].astype(np.float32)
            ct = np.clip((ct - self.ct_min)/(self.ct_max - self.ct_min + 1e-9), 0, 1)
            comb = np.stack([ct]*3, axis=-1)
        else: comb = np.zeros((self.ct_volume.shape[1], self.ct_volume.shape[2], 3), dtype=np.float32)

        # 2. PET
        if self.pet_array is not None and self.show_pet:
            pet = self.pet_array[self.current_slice].astype(np.float32)
            pet = np.clip((pet - self.pet_min)/(self.pet_max - self.pet_min + 1e-9), 0, 1)
            comb = comb * (1 - self.pet_opacity) + self.pet_cmap(pet)[..., :3] * self.pet_opacity

        # 3. 2D Result (Cyan)
        if self.segmentation_mask is not None and self.show_seg:
            sl = self.segmentation_mask[self.current_slice]
            if np.any(sl > 0):
                color = np.array([0.0, 1.0, 1.0]) # Cyan
                comb = np.where(np.stack([sl>0]*3, axis=-1), comb*(1-self.seg_opacity) + color*self.seg_opacity, comb)

        # 4. Volumetric Result (Purple)
        if self.volumetric_mask is not None and self.show_volumetric:
            sl = self.volumetric_mask[self.current_slice]
            if np.any(sl > 0):
                color = np.array([0.8, 0.4, 1.0]) # Purple
                # Draw ON TOP of single slice if they overlap
                comb = np.where(np.stack([sl>0]*3, axis=-1), comb*(1-self.seg_opacity) + color*self.seg_opacity, comb)

        # 5. User Seed (Red)
        if self.seed_mask is not None and self.show_seeds:
            sl = self.seed_mask[self.current_slice]
            if np.any(sl > 0):
                comb = np.where(np.stack([sl>0]*3, axis=-1), np.array([1.0, 0.0, 0.0]), comb)

        qimg = QImage((np.clip(comb,0,1)*255).astype(np.uint8), comb.shape[1], comb.shape[0], comb.shape[1]*3, QImage.Format_RGB888)
        self.label.set_qimage(qimg)

    # --- MOUSE EVENTS ---
    def wheelEvent(self, e): 
        if self.ct_volume is not None:
            self.current_slice = np.clip(self.current_slice + (1 if e.angleDelta().y() > 0 else -1), 0, self.ct_volume.shape[0]-1)
            self.update_slice()
    
    def mousePressEvent(self, e):
        if self.ct_volume is None: return 
        self.dragging = True; self.last_mouse_pos = e.pos()
        if e.button() == Qt.RightButton: self.active_action = "ZOOM"
        elif e.button() == Qt.MiddleButton: self.active_action = "PAN"
        elif e.button() == Qt.LeftButton:
            if self.rad_draw.isChecked(): self.active_action = "DRAW"; self.draw(e.pos(), self.current_color_id)
            elif self.rad_erase.isChecked(): self.active_action = "ERASE"; self.draw(e.pos(), 0)
            else: self.active_action = "WL"

    def mouseMoveEvent(self, e):
        if not self.dragging: return
        dx, dy = e.x()-self.last_mouse_pos.x(), e.y()-self.last_mouse_pos.y(); self.last_mouse_pos = e.pos()
        if self.active_action == "ZOOM": self.label.apply_zoom_pan(-dy*0.02, 0, 0)
        elif self.active_action == "PAN": self.label.apply_zoom_pan(0, dx, dy)
        elif self.active_action in ["DRAW", "ERASE"]: self.draw(e.pos(), self.current_color_id if self.active_action=="DRAW" else 0)
        elif self.active_action == "WL":
            if e.modifiers() & Qt.ControlModifier and self.pet_array is not None:
                self.pet_min = max(0, self.pet_min - dy*0.05); self.pet_max = max(self.pet_min+0.1, self.pet_max - dy*0.05 + dx*0.05)
            else:
                self.ct_min = self.ct_min - dy*2 - dx*2; self.ct_max = self.ct_max - dy*2 + dx*2
            self.update_slice()

    def mouseReleaseEvent(self, e): self.dragging = False

    def draw(self, pos, val):
        c = self.label.get_image_coords(self.label.mapFrom(self, pos))
        if c: 
            y, x = np.ogrid[:self.ct_volume.shape[1], :self.ct_volume.shape[2]]
            self.seed_mask[self.current_slice][(x-c[0])**2 + (y-c[1])**2 <= self.spin_size.value()**2] = val
            self.update_slice()