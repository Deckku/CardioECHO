# CardioECHO: Intelligent Data Extraction from Echocardiogram Images and Videos

## Overview
CardioECHO is a desktop application designed to automate the assessment of cardiac function from echocardiogram videos. Developed using Python, PyTorch, and PyQt6, the system leverages deep learning models to segment the Left Ventricle (LV) and calculate critical clinical metrics, including Ejection Fraction (EF), End-Systolic Volume (ESV), and End-Diastolic Volume (EDV).

This repository contains the source code and the pre-trained model weights required to run or build the application.

## Key Features
* **Automated Analysis:** Uses deep learning to process AVI/MP4 echocardiogram videos.
* **Clinical Metrics:** Automatically calculates Ejection Fraction (EF), End-Systolic Volume (ESV), and derives End-Diastolic Volume (EDV).
* **Semantic Segmentation:** Visualizes the Left Ventricle wall motion using frame-by-frame segmentation overlays.
* **Interactive Interface:** A PyQt6-based GUI allows users to navigate frames, inspect results, and play back video with overlays.
* **Manual Correction:** Includes tools for clinicians to manually paint or erase segmentation masks, triggering real-time recalculation of metrics.
* **Reporting:** Exports findings to standardized PDF reports and DICOM files with embedded private tags for AI metrics.

## Installation

### Prerequisites
* Python 3.8 or higher
* CUDA-capable GPU (Recommended for faster inference, but runs on CPU)

### Setup Steps
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/CardioECHO.git](https://github.com/your-username/CardioECHO.git)
    cd CardioECHO
    ```

2.  **Install dependencies:**
    You can install the required libraries using the following command:
    ```bash
    pip install numpy opencv-python torch torchvision pydicom reportlab matplotlib PyQt6
    ```
    
    *Or if you prefer using a file, create a `requirements.txt` with the above libraries and run:*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Verify Model Files:**
    Ensure the following three model files (included in this repository) are located in the root folder with the script:
    * `deeplabv3_resnet50_random.pt`
    * `mc3_esv_small_best.pth`
    * `best.pt`

## Usage

To run the application directly from the Python script:

```bash
python main.py
