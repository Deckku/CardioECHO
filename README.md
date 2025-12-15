# CardioECHO: Precision Cardiology through Artificial Intelligence

## Overview
CardioECHO is a desktop application designed to automate the assessment of cardiac function from echocardiogram videos. Developed using Python, PyTorch, and PyQt6, the system leverages deep learning models to segment the Left Ventricle (LV) and calculate critical clinical metrics, including Ejection Fraction (EF), End-Systolic Volume (ESV), and End-Diastolic Volume (EDV).

This repository contains the source code for the application. Due to the high storage requirements of high-performance deep learning models, the model weights are hosted externally.

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
* CUDA-capable GPU (Recommended for faster inference)
* Internet connection (to download model weights)

### Setup Steps
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/CardioECHO.git](https://github.com/your-username/CardioECHO.git)
    cd CardioECHO
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *OR run:*
    ```bash
    pip install numpy opencv-python torch torchvision pydicom reportlab matplotlib PyQt6
    ```

3.  **Download Model Weights (Required):**
    The application requires three pre-trained model files (~600 MB total).
    
    1.  Download the files from this link: **https://drive.google.com/drive/folders/17QpRCyoG1jDRxe4NjTJuRY2KwkoM5RPy?usp=sharing**
    2.  Place the downloaded files in the **root directory** of this project (the same folder as `main_app_final.py`).
    3.  Ensure the filenames are exactly as follows:
        * `segmentation.pt`
        * `esv.pth`
        * `ef.pt`

## Usage

### Running the Python Script
Once the dependencies are installed and models are placed in the root folder, run:

```bash
python main_app_final.py
