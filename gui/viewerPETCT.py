from preprocessing import preprocessing
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

# --- Deep Learning Imports ---
import torch
from segmentation.model_defs import MODEL_REGISTRY, DEFAULT_MODEL_CONFIG

from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QWidget, QApplication,
    QPushButton, QFileDialog, QSlider, QHBoxLayout, 
    QSpinBox, QRadioButton, QButtonGroup, QGroupBox, 
    QComboBox, QFrame, QMessageBox, QCheckBox, QSizePolicy, QProgressDialog
)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QImage, QPalette, QColor, QFont

# Ensure this import matches your folder structure. 
# If customQlabel.py is in the same folder, remove 'gui.'
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_folder = os.path.join("segmentation", "models")
        
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
        self.setWindowTitle("LesionLoop Breast ")

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
        
        menu_layout = QVBoxLayout(self.left_panel)
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

        seg_layout.addWidget(self.btn_seg_2d)
        seg_layout.addWidget(self.btn_seg_3d)
        seg_layout.addWidget(self.btn_clear)
        self.grp_seg.setLayout(seg_layout)
        menu_layout.addWidget(self.grp_seg)
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
        self.combo_model.clear()
        
        # Ensure directory exists
        if not os.path.exists(self.model_folder): 
            os.makedirs(self.model_folder, exist_ok=True)
            
        # Get all .pth files
        models = [os.path.basename(m) for m in glob.glob(os.path.join(self.model_folder, "*.pth"))]
        
        if models: 
            self.combo_model.addItems(models)
            self.combo_model.setEnabled(True)
            
            # --- START CHANGE: Set Default Model ---
            default_model = "unet_2.pth"
            
            # check if the specific model exists in the list
            index = self.combo_model.findText(default_model)
            if index >= 0:
                self.combo_model.setCurrentIndex(index)
            # --- END CHANGE ---
            
        else: 
            self.combo_model.addItem("No models found")
            self.combo_model.setEnabled(False)

    def load_selected_model(self):
        model_name = self.combo_model.currentText()
        if not model_name or "No models" in model_name: return False
        
        path = os.path.join(self.model_folder, model_name)
        config = MODEL_REGISTRY.get(model_name, DEFAULT_MODEL_CONFIG)
        
        try:
            progress = self.create_progress("Loading Model", f"Loading {model_name}...")
            self.model = config["class"](**config["params"])
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.to(self.device).eval()
            progress.close()
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}"); return False

    # --- PROPAGATION LOGIC ---
    def run_volumetric_propagation_centered(self, start_z_ai, start_seed_2d, progress_dlg, desc_prefix):
        d, h, w = self.ai_ct_norm.shape
        full_mask_ai = np.zeros((d, h, w), dtype=np.uint8)
        
        # --- START SLICE ---
        input_vol, metadata = preprocessing.run_inference_direct_ai_space(
            self.ai_ct_norm, self.ai_pet_norm, start_seed_2d, start_z_ai
        )
        
        with torch.no_grad():
            out = self.model(torch.from_numpy(input_vol).unsqueeze(0).float().to(self.device))
            pred_mask = (torch.sigmoid(out) > 0.5).float().squeeze().cpu().numpy().astype(np.uint8)

        if np.sum(pred_mask) == 0: return full_mask_ai

        start_seg_result = preprocessing.paste_patch_into_slice(pred_mask, metadata)
        full_mask_ai[start_z_ai] = start_seg_result
        
        # --- PROPAGATION ---
        def propagate_direction(direction):
            current_mask = start_seg_result
            current_z = start_z_ai
            
            while True:
                next_z = current_z + direction
                if not (0 <= next_z < d): break
                
                progress_dlg.setLabelText(f"{desc_prefix}: Processing Slice {next_z}...")
                QApplication.processEvents()

                # Inference
                input_vol, meta = preprocessing.run_inference_direct_ai_space(
                    self.ai_ct_norm, self.ai_pet_norm, current_mask, next_z
                )
                
                with torch.no_grad():
                    out = self.model(torch.from_numpy(input_vol).unsqueeze(0).float().to(self.device))
                    pred_next = (torch.sigmoid(out) > 0.5).float().squeeze().cpu().numpy().astype(np.uint8)

                if np.sum(pred_next) == 0: break
                
                # 1. Reconstruct Raw Slice
                next_full_slice_raw = preprocessing.paste_patch_into_slice(pred_next, meta)

                # 2. ACTIVE FILTERING (Replaces the old check)
                # Removes islands, keeps connected parts.
                next_full_slice_clean = preprocessing.filter_disconnected_blobs(current_mask, next_full_slice_raw)
                
                # If everything was filtered out (tumor ended), stop.
                if np.sum(next_full_slice_clean) == 0:
                    break
                
                full_mask_ai[next_z] = next_full_slice_clean
                current_z = next_z
                current_mask = next_full_slice_clean

        propagate_direction(1)
        propagate_direction(-1)
            
        return full_mask_ai

    # --- BUTTON 1: SINGLE SLICE ---
    def run_segmentation_2d(self):
        if self.ai_ct_norm is None: QMessageBox.warning(self, "Error", "Data not ready."); return
        if self.seed_mask is None or not np.any(self.seed_mask): QMessageBox.warning(self, "Error", "Draw seed."); return
        if not self.load_selected_model(): return

        try:
            print("Running Single Slice Segmentation...")
            input_vol, metadata = preprocessing.run_inference_on_preprocessed(
                self.ai_ct_norm, self.ai_pet_norm, self.ai_scale_info,
                self.seed_mask[self.current_slice], self.current_slice
            )
            
            with torch.no_grad():
                out = self.model(torch.from_numpy(input_vol).unsqueeze(0).float().to(self.device))
                pred_single = (torch.sigmoid(out) > 0.5).float().squeeze().cpu().numpy().astype(np.uint8)
                
            final_single_vol = preprocessing.reconstruct_and_resample_back_optimized(
                pred_single, metadata, self.sitk_ct
            )
            
            self.segmentation_mask = final_single_vol.astype(np.uint8) # Update 2D mask only
            self.update_slice()
            QMessageBox.information(self, "Done", "2D Segmentation Complete (Cyan).")
            
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Error", f"{e}")

    # --- BUTTON 2: VOLUMETRIC ---
    def run_segmentation_3d(self):
        if self.ai_ct_norm is None: QMessageBox.warning(self, "Error", "Data not ready."); return
        if self.seed_mask is None or not np.any(self.seed_mask): QMessageBox.warning(self, "Error", "Draw seed."); return
        if not self.load_selected_model(): return

        try:
            progress = self.create_progress("Volumetric Segmentation", "Starting Pipeline...")
            
            # --- PHASE 1: EXPLORATION ---
            progress.setLabelText("Phase 1: Exploration...")
            QApplication.processEvents()

            z_ai = int(round(self.current_slice * self.ai_scale_info['scale_z']))
            
            from skimage.transform import resize
            h, w = self.ai_ct_norm.shape[1], self.ai_ct_norm.shape[2]
            ai_seed_2d = resize(self.seed_mask[self.current_slice], (h, w), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            
            vol_mask_1 = self.run_volumetric_propagation_centered(z_ai, ai_seed_2d, progress, "Phase 1")
            
            if np.sum(vol_mask_1) == 0:
                progress.close()
                QMessageBox.warning(self, "Result", "Model predicted empty segmentation in Phase 1.")
                return

            # --- PHASE 2: FIND SUV MAX ---
            progress.setLabelText("Phase 2: Finding SUV Max...")
            QApplication.processEvents()
            
            suv_coords = preprocessing.find_suv_max_centroid(self.ai_pet_norm, vol_mask_1)
            
            if suv_coords is None:
                final_ai_mask = vol_mask_1
            else:
                max_z, max_y, max_x = suv_coords
                print(f"Global SUV Max found at AI: {suv_coords}")
                
                # --- PHASE 3: ROBUST PROPAGATION ---
                progress.setLabelText("Phase 3: Robust Propagation...")
                QApplication.processEvents()

                new_seed_2d = preprocessing.create_local_seed_mask(
                    self.ai_ct_norm.shape, (max_z, max_y, max_x), radius=3
                )
                
                final_ai_mask = self.run_volumetric_propagation_centered(max_z, new_seed_2d, progress, "Phase 3")
                # DEBUG: Print total pixels before cleanup
                print(f"DEBUG: Pixels in AI mask before cleanup: {np.sum(final_ai_mask)}")

            # --- PHASE 4: 3D CONNECTIVITY CLEANUP ---
            # Remove any floating islands that might have occurred despite overlap checks
            progress.setLabelText("Phase 4: Cleaning 3D Volume...")
            QApplication.processEvents()
            
            cleaned_ai_mask = preprocessing.keep_largest_connected_component_3d(final_ai_mask)
            #BYPASS CLEANUP
            # cleaned_ai_mask = final_ai_mask

            # --- PHASE 5: RESAMPLE ---
            progress.setLabelText("Finalizing...")
            QApplication.processEvents()
            self.volumetric_mask = preprocessing.resample_full_mask_back_to_viewer(
                cleaned_ai_mask, self.ai_scale_info['sitk_resampled_ref'], self.sitk_ct
            ).astype(np.uint8)

            progress.close()
            self.update_slice()
            QMessageBox.information(self, "Done", "Volumetric Segmentation Complete.\n(IoU Check & 3D Connectivity Applied)")

        except Exception as e:
            if 'progress' in locals(): progress.close()
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Error", f"{e}")

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

    # --- MOUSE EVENTS (FIXED) ---
    def wheelEvent(self, e): 
        if self.ct_volume is not None:
            self.current_slice = np.clip(self.current_slice + (1 if e.angleDelta().y() > 0 else -1), 0, self.ct_volume.shape[0]-1)
            self.update_slice()
    
    def mousePressEvent(self, e):
        # FIX: Check explicitly for None to avoid NumPy boolean error
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