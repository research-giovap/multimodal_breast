import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor

def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel."""
    ax = np.arange(-size//2 + 1., size//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.max(kernel)
    return kernel

class ImageWidget(QWidget):
    def __init__(self, width_img, height_img, scale, image, patch_size, spread_sigma):
        super().__init__()
        self.width_img = width_img
        self.height_img = height_img
        self.scale = scale
        self.image = image
        self.patch_size = patch_size
        self.spread_sigma = spread_sigma

        self.setFixedSize(self.width_img * self.scale, self.height_img * self.scale)
        self.canvas = np.zeros_like(self.image, dtype=float)  # full lesion preview
        self.mouse_pos = None
        self.simulated_lag = 120

        self.setMouseTracking(True)
        self.gaussian = gaussian_kernel(self.patch_size, self.spread_sigma)

    def mouseMoveEvent(self, event):
        x = int(event.x() / self.scale)
        y = int(event.y() / self.scale)
        if 0 <= x < self.width_img and 0 <= y < self.height_img:
            self.mouse_pos = (x, y)

    def patch_predict(self, x, y):
        """Simulate patch-based model prediction (thresholding)."""
        time.sleep(self.simulated_lag / 1000.0)  # simulate GPU inference
        half = self.patch_size // 2
        patch = np.zeros((self.patch_size, self.patch_size))
        for dy in range(-half, half):
            for dx in range(-half, half):
                px = x + dx
                py = y + dy
                if 0 <= px < self.width_img and 0 <= py < self.height_img:
                    patch[dy+half, dx+half] = 1.0 if self.image[py, px] > 128 else 0.0
        return patch

    def merge_patch(self, patch, x, y):
        """Spread patch into canvas using Gaussian kernel."""
        half = self.patch_size // 2
        for dy in range(-half, half):
            for dx in range(-half, half):
                px = x + dx
                py = y + dy
                if 0 <= px < self.width_img and 0 <= py < self.height_img:
                    self.canvas[py, px] = max(
                        self.canvas[py, px],
                        patch[dy+half, dx+half] * self.gaussian[dy+half, dx+half]
                    )

    def segmentation_loop(self):
        if self.mouse_pos:
            x, y = self.mouse_pos
            patch = self.patch_predict(x, y)
            self.merge_patch(patch, x, y)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        for row in range(self.height_img):
            for col in range(self.width_img):
                intensity = int(self.image[row, col])
                painter.fillRect(col*self.scale, row*self.scale, self.scale, self.scale,
                                 QColor(intensity, intensity, intensity))
                if self.canvas[row, col] > 0.1:
                    alpha = int(self.canvas[row, col] * 255)
                    painter.fillRect(col*self.scale, row*self.scale, self.scale, self.scale,
                                     QColor(255, 0, 0, alpha))

class SegDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Patch-Based Segmentation Preview")
        self.width_img = 64
        self.height_img = 64
        self.scale = 8

        # synthetic lesion
        circle_radius = 6
        image = np.full((self.height_img, self.width_img), 50, dtype=np.uint8)
        cx, cy = self.width_img // 2, self.height_img // 2
        for y in range(self.height_img):
            for x in range(self.width_img):
                if (x - cx)**2 + (y - cy)**2 < circle_radius**2:
                    image[y, x] = 200

        patch_size = 16
        spread_sigma = 8  # Gaussian spread for preview

        self.image_widget = ImageWidget(self.width_img, self.height_img, self.scale,
                                        image, patch_size, spread_sigma)

        # Slider for simulated lag
        self.slider_label = QLabel(f"Simulated Lag: 120 ms")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(50)
        self.slider.setMaximum(200)
        self.slider.setValue(120)
        self.slider.valueChanged.connect(self.update_lag)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_widget)
        layout.addWidget(self.slider_label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        # Timer for 10Hz interactive loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.image_widget.segmentation_loop)
        self.timer.start(100)

    def update_lag(self, value):
        self.image_widget.simulated_lag = value
        self.slider_label.setText(f"Simulated Lag: {value} ms")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = SegDemo()
    demo.show()
    sys.exit(app.exec_())
