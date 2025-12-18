from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor

class AspectLabel(QLabel):
    """
    A smart QLabel that handles aspect-ratio preserved scaling, 
    zooming, panning, and coordinate mapping.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pixmap = None
        self.aspect_ratio = 1.0
        
        # Transformation state
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # UI Settings
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)

    def set_qimage(self, qimage: QImage):
        """Sets the QImage and updates the internal pixmap."""
        if qimage.isNull():
            self._pixmap = None
        else:
            self._pixmap = QPixmap.fromImage(qimage)
            if self._pixmap.height() > 0:
                self.aspect_ratio = self._pixmap.width() / self._pixmap.height()
        self.update() # Trigger a repaint

    def apply_zoom_pan(self, zoom_delta, pan_delta_x, pan_delta_y):
        """Updates transform based on user input."""
        # Update Zoom
        self.zoom = max(0.1, min(self.zoom + zoom_delta, 50.0))
        
        # Update Pan
        self.pan_x += pan_delta_x
        self.pan_y += pan_delta_y
        
        self.update()

    def reset_view(self):
        """Resets zoom and pan to default."""
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def get_image_coords(self, widget_pos):
        """
        Maps a widget coordinate (e.g., mouse click) to image pixel coordinates.
        Returns (x, y) if inside image, else None.
        """
        if self._pixmap is None:
            return None

        # Re-calculate the drawing rect (logic must match paintEvent)
        w, h = self.width(), self.height()
        if w == 0 or h == 0: return None
        
        # 1. Base scale to fit window
        scale_w = w / self._pixmap.width()
        scale_h = h / self._pixmap.height()
        base_scale = min(scale_w, scale_h)

        # 2. Apply total scale (Base * Zoom)
        final_scale = base_scale * self.zoom

        # 3. Calculate centered position
        drawn_w = self._pixmap.width() * final_scale
        drawn_h = self._pixmap.height() * final_scale
        
        # 4. Apply Pan offsets
        x_offset = (w - drawn_w) / 2 + self.pan_x
        y_offset = (h - drawn_h) / 2 + self.pan_y

        # 5. Inverse map: (Widget - Offset) / Scale
        img_x = (widget_pos.x() - x_offset) / final_scale
        img_y = (widget_pos.y() - y_offset) / final_scale

        # Check bounds
        if 0 <= img_x < self._pixmap.width() and 0 <= img_y < self._pixmap.height():
            return int(img_x), int(img_y)
        return None

    def paintEvent(self, event):
        """Custom paint to handle zoom/pan and BLACK background."""
        painter = QPainter(self)
        
        # --- FIX: Fill background with black first ---
        painter.fillRect(self.rect(), Qt.black)

        if self._pixmap is None:
            painter.end()
            return

        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        
        # 1. Calculate Base Scale (Fit to Window)
        scale_w = w / self._pixmap.width()
        scale_h = h / self._pixmap.height()
        base_scale = min(scale_w, scale_h)

        # 2. Apply Zoom
        final_scale = base_scale * self.zoom

        # 3. Calculate target dimensions
        target_w = self._pixmap.width() * final_scale
        target_h = self._pixmap.height() * final_scale

        # 4. Calculate positions (Center + Pan)
        x = (w - target_w) / 2 + self.pan_x
        y = (h - target_h) / 2 + self.pan_y

        # Draw
        target_rect = QRectF(x, y, target_w, target_h)
        source_rect = QRectF(0, 0, self._pixmap.width(), self._pixmap.height())
        
        painter.drawPixmap(target_rect, self._pixmap, source_rect)
        painter.end()