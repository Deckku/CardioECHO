# CardioECHO: Intelligent Data Extraction from Echocardiogram Images and Videos

## Overview
CardioECHO is a desktop application designed to automate the assessment of cardiac function from echocardiogram videos. Developed using Python, PyTorch, and PyQt6, the system leverages deep learning models to segment the Left Ventricle (LV) and calculate critical clinical metrics, including Ejection Fraction (EF), End-Systolic Volume (ESV), and End-Diastolic Volume (EDV).

The project is built upon the EchoNet-Dynamic dataset and bridges the gap between research-grade AI models and clinical utility by providing a user-friendly interface with "human-in-the-loop" capabilities for manual correction.

## Key Features
* **Automated Analysis:** Uses deep learning to process AVI/MP4 echocardiogram videos.
* **Clinical Metrics:** Automatically calculates Ejection Fraction (EF), End-Systolic Volume (ESV), and derives End-Diastolic Volume (EDV).
* **Semantic Segmentation:** Visualizes the Left Ventricle wall motion using frame-by-frame segmentation overlays.
* **Interactive Interface:** A PyQt6-based GUI allows users to navigate frames, inspect results, and play back video with overlay.
* **Manual Correction:** Includes tools for clinicians to manually paint or erase segmentation masks, triggering real-time recalculation of metrics.
* **Reporting:** Exports findings to standardized PDF reports and DICOM files with embedded private tags for AI metrics.

## System Architecture

The analysis pipeline consists of three specialized deep learning models and a mathematical derivation step:

1.  **Segmentation (DeepLabV3-ResNet50):** Performs pixel-wise classification to delineate the Left Ventricle in every frame.
2.  **ESV Prediction (MC3 Network):** A spatiotemporal 3D Convolutional Neural Network that estimates the minimum volume of the heart.
3.  **EF Prediction (R(2+1)D-18):** A video-based regression model that analyzes motion across the cardiac cycle to predict Ejection Fraction.
4.  **EDV Derivation:** End-Diastolic Volume is mathematically derived from the predicted EF and ESV to ensure physiological consistency.

## Installation

### Prerequisites
* Python 3.8+
* CUDA-capable GPU (Recommended for faster inference)

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/CardioECHO.git](https://github.com/your-username/CardioECHO.git)
    cd CardioECHO
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Download Model Weights:
    Place the following pre-trained weights in the root directory:
    * `deeplabv3_resnet50_random.pt`
    * `mc3_esv_small_best.pth`
    * `best.pt` (EF Model)

## Usage

### Running from Source
To launch the application using the Python interpreter:

```bash
python main.py
