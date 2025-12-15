import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import pydicom
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime

# PDF Report Imports
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QProgressBar, QTabWidget, 
    QFileDialog, QScrollArea, QFrame, QGridLayout, 
    QSlider, QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox,
    QGroupBox, QDialog, QTextEdit, QSizePolicy, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QPoint
from PyQt6.QtGui import QImage, QPixmap, QIcon, QFont, QPalette, QColor, QPainter, QPen
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

# =============================================================================
# RESOURCE PATH HELPER (REQUIRED FOR .EXE)
# =============================================================================
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# =============================================================================
# DEFAULT CONFIGURATION (Updated to use resource_path)
# =============================================================================
# The app will now look for these specific filenames in the same folder as the exe
DEFAULT_SEG_PATH = resource_path("segmentation.pt")
DEFAULT_ESV_PATH = resource_path("esv.pth")
DEFAULT_EF_PATH = resource_path("ef.pt")

# =============================================================================
# CUSTOM WIDGETS
# =============================================================================
class PaintLabel(QLabel):
    """A QLabel that allows drawing on the image for mask correction."""
    mask_changed = pyqtSignal()

    def __init__(self, width, height, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setScaledContents(True)
        self.setMouseTracking(True)
        self.drawing = False
        self.erase_mode = False
        self.brush_size = 5
        self.current_mask = None # Reference to the 112x112 numpy mask
        self.original_size = (112, 112)

    def set_mask_reference(self, mask_array):
        self.current_mask = mask_array

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.erase_mode = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.drawing = True
            self.erase_mode = True
        self._draw(event.pos())

    def mouseMoveEvent(self, event):
        if self.drawing:
            self._draw(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() in [Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton]:
            self.drawing = False
            self.mask_changed.emit()

    def _draw(self, pos):
        if self.current_mask is None: return

        # Map widget coordinates (200x200) to mask coordinates (112x112)
        x_ratio = self.original_size[0] / self.width()
        y_ratio = self.original_size[1] / self.height()
        
        mx = int(pos.x() * x_ratio)
        my = int(pos.y() * y_ratio)

        # Draw on numpy array
        val = 0 if self.erase_mode else 255
        radius = int(self.brush_size * x_ratio)
        
        cv2.circle(self.current_mask, (mx, my), radius, (val), -1)
        self.update() # Trigger repaint in parent logic

# =============================================================================
# MODEL DEFINITIONS (Unchanged)
# =============================================================================
class MC3BlockSmall(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3),
                      stride=(1,stride,stride), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        self.temporal = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(3,1,1),
                      stride=(stride,1,1), padding=(1,0,0), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=(stride,stride,stride), bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.spatial(x)
        out = self.temporal(out)
        if self.downsample:
            identity = self.downsample(identity)
        return torch.relu(out + identity)

class MC3NetSmall(nn.Module):
    def __init__(self, num_frames=8, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3,7,7), stride=(1,2,2),
                      padding=(1,3,3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1,3,3), stride=(1,2,2), padding=(0,1,1))
        )
        self.layer1 = self._make_layer(32, 32, 1)
        self.layer2 = self._make_layer(32, 64, 1, stride=2)
        self.layer3 = self._make_layer(64, 128, 1, stride=2)
        self.layer4 = self._make_layer(128, 256, 1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256,1)
        )

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = [MC3BlockSmall(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(MC3BlockSmall(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv_img.copy()
    if len(rgb_image.shape) == 2:
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
    
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(convert_to_Qt_format)

def calculate_lv_area(mask, pixel_spacing=0.154):
    area_pixels = np.sum(mask > 127)
    area_mm2 = area_pixels * (pixel_spacing ** 2)
    return area_mm2

def overlay_segmentation(frame, mask, alpha=0.5, color=(0, 255, 0)):
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    overlay = frame.copy()
    overlay[mask_resized > 127] = color
    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    return result

def get_contour(mask):
    mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    return max(contours, key=cv2.contourArea)

def overlay_contour(frame, contour, color=(255, 0, 0), thickness=2):
    frame_disp = frame.copy()
    if contour is not None:
        cv2.drawContours(frame_disp, [contour], -1, color, thickness)
    return frame_disp

# =============================================================================
# WORKER THREAD FOR ANALYSIS
# =============================================================================
class AnalysisWorker(QThread):
    progress_updated = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path, model_paths, device, max_frames, pixel_spacing):
        super().__init__()
        self.video_path = video_path
        self.model_paths = model_paths
        self.device = device
        self.max_frames = max_frames
        self.pixel_spacing = pixel_spacing
        self.is_running = True

    def run(self):
        try:
            # 1. Load Models
            self.progress_updated.emit(5, "Loading models...")
            
            # Seg Model
            seg_model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=False)
            seg_model.classifier[-1] = nn.Conv2d(seg_model.classifier[-1].in_channels, 1, kernel_size=seg_model.classifier[-1].kernel_size)
            if os.path.exists(self.model_paths['seg']):
                ckpt = torch.load(self.model_paths['seg'], map_location=self.device, weights_only=False)
                state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                seg_model.load_state_dict(new_state_dict)
            seg_model.to(self.device).eval()

            # ESV Model
            esv_model = MC3NetSmall(num_frames=8).to(self.device)
            if os.path.exists(self.model_paths['esv']):
                ckpt = torch.load(self.model_paths['esv'], map_location=self.device, weights_only=False)
                state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
                esv_model.load_state_dict(state_dict)
            esv_model.eval()

            # EF Model
            ef_model = torchvision.models.video.r2plus1d_18(pretrained=False)
            ef_model.fc = nn.Linear(ef_model.fc.in_features, 1)
            if os.path.exists(self.model_paths['ef']):
                ckpt = torch.load(self.model_paths['ef'], map_location=self.device, weights_only=False)
                state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                ef_model.load_state_dict(new_state_dict)
            ef_model.to(self.device).eval()

            # 2. Process Video
            self.progress_updated.emit(15, "Processing frames for segmentation...")
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            limit = self.max_frames if self.max_frames else total_frames
            
            frames, frames_resized, segmentations, overlays, areas = [], [], [], [], []
            count = 0

            while cap.isOpened() and self.is_running:
                ret, frame = cap.read()
                if not ret or count >= limit:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized_img = cv2.resize(frame_rgb, (112, 112))
                
                frame_norm = frame_resized_img.astype(np.float32) / 255.0
                frame_tensor = torch.from_numpy(frame_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    out = seg_model(frame_tensor)['out']
                    seg = torch.sigmoid(out).cpu().numpy()[0, 0]
                    mask = (seg > 0.5).astype(np.uint8) * 255

                area = calculate_lv_area(mask, self.pixel_spacing)
                overlay = overlay_segmentation(frame_resized_img, mask)

                frames.append(frame_rgb) 
                frames_resized.append(frame_resized_img)
                segmentations.append(mask)
                overlays.append(overlay)
                areas.append(area)

                count += 1
                prog = 15 + int((count / limit) * 50) 
                if count % 5 == 0:
                    self.progress_updated.emit(prog, f"Segmenting frame {count}/{limit}")

            cap.release()

            # 3. Analyze Cycles
            if len(areas) > 2:
                ed_frame = int(np.argmax(areas))
                es_frame = int(np.argmin(areas))
            else:
                ed_frame, es_frame = 0, 0
            
            contour_ed = get_contour(segmentations[ed_frame])
            contour_es = get_contour(segmentations[es_frame])

            # 4. Predict Metrics
            self.progress_updated.emit(70, "Predicting ESV...")
            pred_esv = self._predict_esv_val(esv_model)
            
            self.progress_updated.emit(85, "Predicting EF...")
            pred_ef = self._predict_ef_val(ef_model)

            # 5. Deduce EDV
            calc_edv = None
            try:
                if pred_ef < 100 and pred_ef >= 0:
                    calc_edv = pred_esv / (1.0 - (pred_ef / 100.0))
            except:
                pass

            results = {
                'frames': frames_resized, 
                'raw_frames': frames,
                'segmentations': segmentations,
                'overlays': overlays,
                'areas': areas,
                'ed_frame': ed_frame,
                'es_frame': es_frame,
                'contour_ed': contour_ed,
                'contour_es': contour_es,
                'pred_esv': pred_esv,
                'pred_ef': pred_ef,
                'calculated_edv': calc_edv
            }
            
            self.progress_updated.emit(100, "Analysis Complete")
            self.finished.emit(results)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _predict_esv_val(self, model):
        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total-1, min(32, total), dtype=int)[::4][:8]
        
        frames_list = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                f = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (64,64))
                frames_list.append(f)
            else:
                frames_list.append(np.zeros((64,64,3), np.uint8))
        cap.release()
        while len(frames_list) < 8:
            frames_list.append(frames_list[-1])
        inp = np.stack(frames_list) / 255.0
        inp = (inp - 0.5) / 0.5
        inp = torch.FloatTensor(inp).permute(3,0,1,2).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return model(inp).item()

    def _predict_ef_val(self, model):
        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, total-1, 32).astype(int)
        frames = []
        for i in range(total):
            ret, frame = cap.read()
            if not ret: break
            if i in idxs:
                f = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (112,112))
                frames.append(f)
        cap.release()
        while len(frames) < 32: frames.append(frames[-1])
        tens = torch.tensor(np.array(frames), dtype=torch.float32).permute(3,0,1,2) / 255.0
        tens = tens.unsqueeze(0).to(self.device)
        with torch.no_grad():
            return model(tens).item()
    
    def stop(self):
        self.is_running = False

# =============================================================================
# MAIN WINDOW
# =============================================================================
class EchoNetApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CardioECHO")
        self.setGeometry(100, 100, 1400, 900)

        self.current_video_path = None
        self.analysis_results = None
        self.worker = None
        self.dark_mode = True # Default

        # Video Playback Variables
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.is_playing = False
        self.frame_rate = 30 # Default assumed

        self.setup_ui()
        self.apply_theme()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- Sidebar ---
        sidebar = QFrame()
        sidebar.setFrameShape(QFrame.Shape.StyledPanel)
        sidebar.setFixedWidth(320)
        sidebar_layout = QVBoxLayout(sidebar)
        
        # Header
        title_lbl = QLabel("CardioECHO")
        title_lbl.setStyleSheet("font-size: 20px; font-weight: bold; color: #E74C3C;")
        sidebar_layout.addWidget(title_lbl)

        # Theme Toggle
        self.btn_theme = QPushButton("Toggle Light/Dark Mode")
        self.btn_theme.clicked.connect(self.toggle_theme)
        sidebar_layout.addWidget(self.btn_theme)

        # Configuration
        conf_group = QGroupBox("Configuration")
        conf_layout = QVBoxLayout()
        
        self.device_lbl = QLabel("Device: " + ("CUDA" if torch.cuda.is_available() else "CPU"))
        conf_layout.addWidget(self.device_lbl)

        self.path_seg = self.create_path_input("Seg Model Path", DEFAULT_SEG_PATH)
        self.path_esv = self.create_path_input("ESV Model Path", DEFAULT_ESV_PATH)
        self.path_ef = self.create_path_input("EF Model Path", DEFAULT_EF_PATH)
        
        conf_layout.addWidget(QLabel("Segmentation Model:"))
        conf_layout.addLayout(self.path_seg)
        conf_layout.addWidget(QLabel("ESV Model:"))
        conf_layout.addLayout(self.path_esv)
        conf_layout.addWidget(QLabel("EF Model:"))
        conf_layout.addLayout(self.path_ef)

        # Processing Options
        conf_layout.addWidget(QLabel("Max Frames (0=All):"))
        self.spin_max_frames = QSpinBox()
        self.spin_max_frames.setRange(0, 1000)
        self.spin_max_frames.setValue(100)
        conf_layout.addWidget(self.spin_max_frames)

        conf_layout.addWidget(QLabel("Pixel Spacing (mm):"))
        self.spin_pixel = QDoubleSpinBox()
        self.spin_pixel.setRange(0.01, 1.0)
        self.spin_pixel.setSingleStep(0.001)
        self.spin_pixel.setValue(0.154)
        conf_layout.addWidget(self.spin_pixel)

        conf_group.setLayout(conf_layout)
        sidebar_layout.addWidget(conf_group)
        sidebar_layout.addStretch()
        
        main_layout.addWidget(sidebar)

        # --- Main Tabs ---
        self.tabs = QTabWidget()
        self.tab_analysis = QWidget()
        self.tab_about = QWidget()
        
        self.tabs.addTab(self.tab_analysis, "Analysis")
        self.tabs.addTab(self.tab_about, "About")
        
        main_layout.addWidget(self.tabs)
        self.setup_analysis_tab()
        self.setup_about_tab()

    def create_path_input(self, placeholder, default):
        layout = QHBoxLayout()
        line_edit = QLineEdit(default)
        line_edit.setPlaceholderText(placeholder)
        line_edit.setObjectName(placeholder) 
        btn = QPushButton("...")
        btn.setFixedWidth(30)
        btn.clicked.connect(lambda: self.browse_file(line_edit, "Model File (*.pt *.pth)"))
        layout.addWidget(line_edit)
        layout.addWidget(btn)
        return layout

    def browse_file(self, line_edit_widget, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter_str)
        if path:
            line_edit_widget.setText(path)

    def setup_analysis_tab(self):
        layout = QVBoxLayout(self.tab_analysis)
        
        # Patient Info
        self.patient_group = QGroupBox("Patient Information")
        info_layout = QGridLayout()
        self.txt_p_name = QLineEdit(); info_layout.addWidget(QLabel("Name:"), 0, 0); info_layout.addWidget(self.txt_p_name, 0, 1)
        self.txt_p_id = QLineEdit(); info_layout.addWidget(QLabel("ID:"), 0, 2); info_layout.addWidget(self.txt_p_id, 0, 3)
        self.cb_sex = QComboBox(); self.cb_sex.addItems(["", "M", "F"]); info_layout.addWidget(QLabel("Sex:"), 0, 4); info_layout.addWidget(self.cb_sex, 0, 5)
        self.txt_age = QLineEdit(); info_layout.addWidget(QLabel("Age:"), 1, 0); info_layout.addWidget(self.txt_age, 1, 1)
        self.txt_study = QLineEdit("CardioECHO Analysis"); info_layout.addWidget(QLabel("Study Desc:"), 1, 2); info_layout.addWidget(self.txt_study, 1, 3)
        self.patient_group.setLayout(info_layout)
        layout.addWidget(self.patient_group)

        # Video Upload
        vid_layout = QHBoxLayout()
        self.btn_load_video = QPushButton("Upload Echocardiogram Video")
        self.btn_load_video.clicked.connect(self.load_video)
        self.lbl_video_path = QLabel("No video loaded")
        vid_layout.addWidget(self.btn_load_video)
        vid_layout.addWidget(self.lbl_video_path)
        layout.addLayout(vid_layout)

        # Run Button
        self.btn_run = QPushButton("Run Full Cardiac Analysis")
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet("background-color: #3498DB; color: white; font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self.run_analysis)
        layout.addWidget(self.btn_run)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.lbl_status = QLabel("")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.lbl_status)

        # Results Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVisible(False) # Hidden by default
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.scroll_area.setWidget(self.results_widget)
        layout.addWidget(self.scroll_area)

        # --- Inside Results Widget ---
        
        # Metrics
        metrics_group = QGroupBox("Final Cardiac Metrics")
        metrics_layout = QHBoxLayout()
        self.lbl_esv = QLabel("ESV: --")
        self.lbl_esv.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.lbl_ef = QLabel("EF: --")
        self.lbl_ef.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.lbl_edv = QLabel("EDV: --")
        self.lbl_edv.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.lbl_clinical = QLabel("")
        metrics_layout.addWidget(self.lbl_esv)
        metrics_layout.addWidget(self.lbl_ef)
        metrics_layout.addWidget(self.lbl_edv)
        metrics_layout.addWidget(self.lbl_clinical)
        metrics_group.setLayout(metrics_layout)
        self.results_layout.addWidget(metrics_group)

        # Landmarks
        land_group = QGroupBox("Cardiac Cycle Landmarks")
        land_layout = QHBoxLayout()
        self.lbl_ed_img = QLabel("ED Frame"); self.lbl_ed_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_es_img = QLabel("ES Frame"); self.lbl_es_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        land_layout.addWidget(self.lbl_ed_img)
        land_layout.addWidget(self.lbl_es_img)
        land_group.setLayout(land_layout)
        self.results_layout.addWidget(land_group)

        # Plot
        self.figure = plt.figure(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(400)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.results_layout.addWidget(self.canvas)

        # Frame Browser
        browser_group = QGroupBox("Interactive Frame Browser")
        browser_layout = QVBoxLayout()
        
        # Images Row
        imgs_layout = QHBoxLayout()
        self.lbl_brow_orig = QLabel("Original"); self.lbl_brow_orig.setFixedSize(200, 200); self.lbl_brow_orig.setScaledContents(True)
        
        # === CUSTOM PAINT LABEL FOR MASK ===
        self.lbl_brow_seg = PaintLabel(200, 200)
        self.lbl_brow_seg.mask_changed.connect(self.on_mask_edited) # Connect signal
        
        self.lbl_brow_over = QLabel("Overlay"); self.lbl_brow_over.setFixedSize(200, 200); self.lbl_brow_over.setScaledContents(True)
        
        imgs_layout.addWidget(self.lbl_brow_orig)
        imgs_layout.addWidget(self.lbl_brow_seg)
        imgs_layout.addWidget(self.lbl_brow_over)
        
        # Info & Instructions
        self.lbl_frame_info = QLabel("Frame: 0")
        lbl_instr = QLabel("Left Click to Draw (White) | Right Click to Erase (Black)")
        lbl_instr.setStyleSheet("font-style: italic; color: gray;")
        
        browser_layout.addLayout(imgs_layout)
        browser_layout.addWidget(self.lbl_frame_info)
        browser_layout.addWidget(lbl_instr)
        
        # Recalculate Button
        self.btn_recalc = QPushButton("Recalculate Areas from Mask Edits")
        self.btn_recalc.clicked.connect(self.recalculate_areas)
        browser_layout.addWidget(self.btn_recalc)
        
        # Slider & Video Controls
        self.slider_frame = QSlider(Qt.Orientation.Horizontal)
        self.slider_frame.valueChanged.connect(self.update_browser_image)
        browser_layout.addWidget(self.slider_frame)

        ctrl_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.cb_speed = QComboBox()
        self.cb_speed.addItems(["0.5x", "1.0x", "2.0x"])
        self.cb_speed.setCurrentIndex(1)
        self.cb_speed.currentTextChanged.connect(self.change_speed)
        
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(QLabel("Speed:"))
        ctrl_layout.addWidget(self.cb_speed)
        ctrl_layout.addStretch()
        
        browser_layout.addLayout(ctrl_layout)
        
        browser_group.setLayout(browser_layout)
        self.results_layout.addWidget(browser_group)

        # Exports
        export_layout = QHBoxLayout()
        btn_export_dicom = QPushButton("Export ED to DICOM")
        btn_export_dicom.clicked.connect(self.export_dicom)
        btn_export_pdf = QPushButton("Export PDF Report")
        btn_export_pdf.clicked.connect(self.export_pdf_report)
        btn_view = QPushButton("View External DICOM")
        btn_view.clicked.connect(self.view_dicom)
        export_layout.addWidget(btn_export_dicom)
        export_layout.addWidget(btn_export_pdf)
        export_layout.addWidget(btn_view)
        self.results_layout.addLayout(export_layout)

    def setup_about_tab(self):
        layout = QVBoxLayout(self.tab_about)
        text = QTextEdit()
        text.setReadOnly(True)
        html_content = """
        <h2>CardioECHO</h2>
        <p><b>Created by:</b> Azami Hassani Adnane<br>
        <b>Supervised by:</b> Pr. Mohamed Hosni</p>
        
        <h3>1. Clinical Metrics Explained</h3>
        <ul>
        <li><b>LV (Left Ventricle):</b> The main pumping chamber of the heart.</li>
        <li><b>ESV (End-Systolic Volume):</b> The volume of blood in the LV at the end of contraction (systole). Lower is usually better.</li>
        <li><b>EDV (End-Diastolic Volume):</b> The volume of blood in the LV just before contraction (diastole).</li>
        <li><b>EF (Ejection Fraction):</b> The percentage of blood leaving the heart each time it squeezes.</li>
        </ul>
        
        <h3>2. Ejection Fraction (EF) Ranges</h3>
        <p><b>Male:</b></p>
        <ul>
            <li><b>Normal:</b> 52 - 70%</li>
            <li><b>Mildly Abnormal:</b> 41 - 51%</li>
            <li><b>Moderately Abnormal:</b> 30 - 40%</li>
            <li><b>Severely Abnormal:</b> less than 30%</li>
        </ul>
        <p><b>Female:</b></p>
        <ul>
            <li><b>Normal:</b> 54 - 74%</li>
            <li><b>Mildly Abnormal:</b> 41 - 53%</li>
            <li><b>Moderately Abnormal:</b> 30 - 40%</li>
            <li><b>Severely Abnormal:</b> less than 30% </li>
        </ul>

        <h3>3. How to Use</h3>
        <ol>
        <li><b>Configuration:</b> Ensure model paths are correct in the sidebar.</li>
        <li><b>Patient Info:</b> Enter patient details and <b>Select Sex (M/F)</b> for accurate diagnosis.</li>
        <li><b>Upload:</b> Click "Upload Echocardiogram Video" to select an AVI/MP4 file.</li>
        <li><b>Run:</b> Click "Run Full Cardiac Analysis". The AI will segment the LV and calculate metrics.</li>
        <li><b>Review:</b> Use the "Frame-by-Frame Browser" to inspect the segmentation.</li>
        <li><b>Edit (Optional):</b> If the mask is inaccurate, Left Click to draw (add) or Right Click to erase on the "Mask" view. Click "Recalculate" to update metrics.</li>
        <li><b>Export:</b> Save results as a PDF Report or DICOM file.</li>
        </ol>
        """
        text.setHtml(html_content)
        layout.addWidget(text)

    # --- THEME LOGIC ---
    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()

    def apply_theme(self):
        palette = QPalette()
        if self.dark_mode:
            palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
            palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
            palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        else:
            palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
            palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
            palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.AlternateBase, QColor(233, 231, 227))
            palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
            palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
            palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
            palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
            palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
            palette.setColor(QPalette.ColorRole.Link, QColor(0, 0, 255))
            palette.setColor(QPalette.ColorRole.Highlight, QColor(76, 163, 224))
            palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        
        QApplication.instance().setPalette(palette)

    # --- VIDEO PLAYBACK LOGIC ---
    def toggle_playback(self):
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("Play")
            self.is_playing = False
        else:
            self.timer.start(int(1000 / self.frame_rate))
            self.btn_play.setText("Pause")
            self.is_playing = True

    def stop_playback(self):
        self.timer.stop()
        self.is_playing = False
        self.btn_play.setText("Play")
        self.slider_frame.setValue(0)

    def change_speed(self, text):
        speed_mult = float(text.replace("x", ""))
        self.frame_rate = 30 * speed_mult
        if self.is_playing:
            self.timer.start(int(1000 / self.frame_rate))

    def next_frame(self):
        if not self.analysis_results: return
        curr = self.slider_frame.value()
        nxt = curr + 1
        if nxt >= len(self.analysis_results['frames']):
            nxt = 0 # Loop
        self.slider_frame.setValue(nxt)

    # --- MASK EDITING LOGIC ---
    def on_mask_edited(self):
        """Called when user releases mouse after drawing"""
        idx = self.slider_frame.value()
        # Update overlay for current frame immediately
        mask = self.analysis_results['segmentations'][idx]
        frame = self.analysis_results['frames'][idx]
        
        # Update overlay in data structure
        new_overlay = overlay_segmentation(frame, mask)
        self.analysis_results['overlays'][idx] = new_overlay
        
        # Update GUI
        self.lbl_brow_over.setPixmap(convert_cv_qt(new_overlay))
        self.lbl_frame_info.setText("Frame: {} | Mask Modified (Click Recalculate)".format(idx))

    def recalculate_areas(self):
        if not self.analysis_results: return
        
        pixel_spacing = self.spin_pixel.value()
        areas = []
        for mask in self.analysis_results['segmentations']:
            areas.append(calculate_lv_area(mask, pixel_spacing))
        
        self.analysis_results['areas'] = areas
        
        # Recalc ED/ES frames based on new areas
        ed_frame = int(np.argmax(areas))
        es_frame = int(np.argmin(areas))
        
        self.analysis_results['ed_frame'] = ed_frame
        self.analysis_results['es_frame'] = es_frame
        self.analysis_results['contour_ed'] = get_contour(self.analysis_results['segmentations'][ed_frame])
        self.analysis_results['contour_es'] = get_contour(self.analysis_results['segmentations'][es_frame])
        
        # Redraw plots and labels
        self.display_results(self.analysis_results) # Re-use display logic
        QMessageBox.information(self, "Recalculated", "Areas and ED/ES frames updated based on mask edits.")

    # --- MAIN LOGIC ---
    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.avi *.mp4 *.mov)")
        if path:
            self.current_video_path = path
            self.lbl_video_path.setText(os.path.basename(path))
            self.btn_run.setEnabled(True)
            self.scroll_area.setVisible(False)
            self.stop_playback()

    def run_analysis(self):
        if not self.current_video_path: return

        paths = {
            'seg': self.findChild(QLineEdit, "Seg Model Path").text(),
            'esv': self.findChild(QLineEdit, "ESV Model Path").text(),
            'ef': self.findChild(QLineEdit, "EF Model Path").text()
        }
        
        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.scroll_area.setVisible(False)

        self.worker = AnalysisWorker(self.current_video_path, paths, 
                                     'cuda' if torch.cuda.is_available() else 'cpu', 
                                     self.spin_max_frames.value(), self.spin_pixel.value())
        self.worker.progress_updated.connect(lambda v, m: (self.progress_bar.setValue(v), self.lbl_status.setText(m)))
        self.worker.error_occurred.connect(lambda m: (QMessageBox.critical(self, "Error", m), self.btn_run.setEnabled(True)))
        self.worker.finished.connect(self.display_results)
        self.worker.start()

    def display_results(self, results):
        self.analysis_results = results
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Done!")
        self.scroll_area.setVisible(True)

        # Metrics
        self.lbl_esv.setText(f"ESV: {results['pred_esv']:.2f} mL")
        self.lbl_ef.setText(f"EF: {results['pred_ef']:.2f}%")
        edv_txt = f"{results['calculated_edv']:.2f} mL" if results['calculated_edv'] else "N/A"
        self.lbl_edv.setText(f"EDV: {edv_txt}")

        # === GENDER-SPECIFIC EF LOGIC ===
        ef = results['pred_ef']
        sex = self.cb_sex.currentText()
        
        status = "Unknown"
        color = "gray"

        if sex == "M":
            if ef >= 52:
                status, color = "Normal", "green"
            elif 41 <= ef < 52:
                status, color = "Mildly Abnormal", "orange"
            elif 30 <= ef < 41:
                status, color = "Moderately Abnormal", "darkorange"
            else:
                status, color = "Severely Abnormal", "red"
        elif sex == "F":
            if ef >= 54:
                status, color = "Normal", "green"
            elif 41 <= ef < 54:
                status, color = "Mildly Abnormal", "orange"
            elif 30 <= ef < 41:
                status, color = "Moderately Abnormal", "darkorange"
            else:
                status, color = "Severely Abnormal", "red"
        else:
             # Default fallback if sex is not selected
             status = "Select Sex for Interpretation"
             color = "gray"

        self.lbl_clinical.setText(status)
        self.lbl_clinical.setStyleSheet(f"color: {color}; font-weight: bold;")

        # Landmarks
        ed_idx, es_idx = results['ed_frame'], results['es_frame']
        ed_img = overlay_contour(results['frames'][ed_idx], results['contour_ed'], (255,0,0))
        es_img = overlay_contour(results['frames'][es_idx], results['contour_es'], (0,0,255))

        self.lbl_ed_img.setPixmap(convert_cv_qt(ed_img).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
        self.lbl_es_img.setPixmap(convert_cv_qt(es_img).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))

        # Plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(results['areas'], linewidth=2, label='LV Area')
        ax.axvline(ed_idx, color='red', linestyle='--', label="ED")
        ax.axvline(es_idx, color='blue', linestyle='--', label="ES")
        ax.set_title("LV Area Dynamics")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Area (mm²)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()

        # Browser
        self.slider_frame.setMaximum(len(results['frames']) - 1)
        self.slider_frame.setValue(ed_idx)
        self.update_browser_image()

    def update_browser_image(self):
        if not self.analysis_results: return
        idx = self.slider_frame.value()
        res = self.analysis_results
        
        self.lbl_frame_info.setText(f"Frame: {idx} | LV Area: {res['areas'][idx]:.2f} mm²")
        
        # Original
        self.lbl_brow_orig.setPixmap(convert_cv_qt(res['frames'][idx]))
        
        # Mask (Paint Label)
        mask = res['segmentations'][idx]
        self.lbl_brow_seg.set_mask_reference(mask) # Pass raw numpy array by ref
        mask_disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        self.lbl_brow_seg.setPixmap(convert_cv_qt(mask_disp))

        # Overlay
        self.lbl_brow_over.setPixmap(convert_cv_qt(res['overlays'][idx]))

    # --- EXPORT LOGIC ---
    def export_dicom(self):
        if not self.analysis_results: return
        path, _ = QFileDialog.getSaveFileName(self, "Save DICOM", "echonet_analysis.dcm", "DICOM (*.dcm)")
        if not path: return

        # 1. Setup Basic Metadata
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = generate_uid()
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # 2. Add Patient Info (Always use text fields now)
        ds.PatientName = self.txt_p_name.text()
        ds.PatientID = self.txt_p_id.text()
        ds.PatientSex = self.cb_sex.currentText()
        ds.PatientAge = self.txt_age.text()
        ds.StudyDescription = self.txt_study.text()
        ds.Modality = "US"
        
        now = datetime.now()
        ds.StudyDate = now.strftime("%Y%m%d")
        ds.StudyTime = now.strftime("%H%M%S")

        # 3. Prepare Image Data (Critical Step)
        ed_idx = self.analysis_results['ed_frame']
        img_rgb = self.analysis_results['frames'][ed_idx]
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        ds.Rows, ds.Columns = img_gray.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.HighBit = 7
        ds.BitsStored = 8
        ds.BitsAllocated = 8
        ds.PixelData = img_gray.tobytes()

        # 4. Add Custom AI Metrics (Private Tags)
        ds.add_new((0x0011, 0x1010), "LO", f"ESV={self.analysis_results['pred_esv']:.2f} mL")
        ds.add_new((0x0011, 0x1011), "LO", f"EF={self.analysis_results['pred_ef']:.2f} %")
        if self.analysis_results['calculated_edv']:
            ds.add_new((0x0011, 0x1012), "LO", f"EDV={self.analysis_results['calculated_edv']:.2f} mL")

        try:
            ds.save_as(path)
            QMessageBox.information(self, "Success", "DICOM exported successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save DICOM: {str(e)}")

    def export_pdf_report(self):
        if not self.analysis_results: return
        path, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", "report.pdf", "PDF Files (*.pdf)")
        if not path: return

        try:
            c = canvas.Canvas(path, pagesize=LETTER)
            width, height = LETTER
            
            # Header
            c.setFont("Helvetica-Bold", 20)
            c.drawString(50, height - 50, "CardioECHO Analysis Report")
            
            # Patient Info (Always show)
            c.setFont("Helvetica", 12)
            y = height - 80
            c.drawString(50, y, f"Patient Name: {self.txt_p_name.text()}")
            c.drawString(300, y, f"Patient ID: {self.txt_p_id.text()}")
            y -= 20
            c.drawString(50, y, f"Age: {self.txt_age.text()}   Sex: {self.cb_sex.currentText()}")
            y -= 20
            
            c.drawString(50, y, f"Study: {self.txt_study.text()}")
            c.drawString(300, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            # Metrics Box
            y -= 40
            c.setStrokeColorRGB(0,0,0)
            c.rect(40, y - 60, 520, 70, fill=0)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(60, y - 20, "Results")
            c.setFont("Helvetica", 12)
            c.drawString(60, y - 45, f"EF: {self.analysis_results['pred_ef']:.2f} %")
            c.drawString(200, y - 45, f"ESV: {self.analysis_results['pred_esv']:.2f} mL")
            if self.analysis_results['calculated_edv']:
                c.drawString(350, y - 45, f"EDV: {self.analysis_results['calculated_edv']:.2f} mL")
            
            # Graph (Fixed spacing)
            y -= 280
            
            # Save matplotlib graph to buffer
            buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            self.figure.tight_layout()
            self.figure.savefig(buf.name, format='png', dpi=300)
            
            c.drawImage(buf.name, 50, y, width=500, height=200)
            buf.close()
            os.unlink(buf.name)
            
            # Images (ED/ES)
            y -= 160
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y + 130, "End-Diastole (ED)")
            c.drawString(300, y + 130, "End-Systole (ES)")
            
            def draw_cv_img(img, x, y):
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(tmp.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                c.drawImage(tmp.name, x, y, width=200, height=200)
                tmp.close()
                os.unlink(tmp.name)

            ed_img = overlay_contour(self.analysis_results['frames'][self.analysis_results['ed_frame']], 
                                     self.analysis_results['contour_ed'], (255,0,0))
            es_img = overlay_contour(self.analysis_results['frames'][self.analysis_results['es_frame']], 
                                     self.analysis_results['contour_es'], (0,0,255))
            
            draw_cv_img(ed_img, 50, y - 80)
            draw_cv_img(es_img, 300, y - 80)
            
            c.save()
            QMessageBox.information(self, "Success", "PDF Report saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save PDF: {str(e)}")

    def view_dicom(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open DICOM", "", "DICOM (*.dcm)")
        if not path: return
        
        try:
            ds = pydicom.dcmread(path)
            
            dlg = QDialog(self)
            dlg.setWindowTitle("DICOM Viewer")
            dlg.resize(600, 500)
            layout = QVBoxLayout(dlg)
            
            if hasattr(ds, "pixel_array"):
                arr = ds.pixel_array.astype(float)
                if arr.max() > arr.min():
                    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
                else:
                    arr = np.zeros_like(arr)
                
                arr = arr.astype(np.uint8)
                if len(arr.shape) == 2:
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                
                lbl = QLabel()
                pixmap = convert_cv_qt(arr)
                lbl.setPixmap(pixmap.scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio))
                layout.addWidget(lbl, alignment=Qt.AlignmentFlag.AlignCenter)
            else:
                layout.addWidget(QLabel("No pixel data found."))

            info = QTextEdit()
            info.setReadOnly(True)
            txt = f"Patient: {getattr(ds, 'PatientName', 'N/A')}\n"
            txt += f"ID: {getattr(ds, 'PatientID', 'N/A')}\n"
            
            found_ai = False
            for elem in ds.iterall():
                if elem.tag.group == 0x0011:
                    txt += f"AI Metric: {elem.value}\n"
                    found_ai = True
            if not found_ai: txt += "\n(No AI metrics found)"
                
            info.setText(txt)
            info.setMaximumHeight(100)
            layout.addWidget(info)
            
            dlg.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read DICOM: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = EchoNetApp()
    window.show()
    sys.exit(app.exec())