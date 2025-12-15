# CardioECHO: Precision Cardiology through Artificial Intelligence

## Overview

CardioECHO is a desktop application designed to automate the assessment of cardiac function from echocardiogram videos. Developed using Python, PyTorch, and PyQt6, the system leverages deep learning models to segment the Left Ventricle (LV) and calculate critical clinical metrics, including Ejection Fraction (EF), End-Systolic Volume (ESV), and End-Diastolic Volume (EDV).

This repository contains the source code for the application. Due to the high storage requirements of high-performance deep learning models, the model weights are hosted externally.

---

## Key Features

* **Automated Analysis:** Uses deep learning to process AVI/MP4 echocardiogram videos
* **Clinical Metrics:** Automatically calculates Ejection Fraction (EF), End-Systolic Volume (ESV), and derives End-Diastolic Volume (EDV)
* **Semantic Segmentation:** Visualizes the Left Ventricle wall motion using frame-by-frame segmentation overlays
* **Interactive Interface:** A PyQt6-based GUI allows users to navigate frames, inspect results, and play back video with overlays
* **Manual Correction:** Includes tools for clinicians to manually paint or erase segmentation masks, triggering real-time recalculation of metrics
* **Reporting:** Exports findings to standardized PDF reports and DICOM files with embedded private tags for AI metrics

---

## Installation

### Prerequisites

* Python 3.8 or higher
* CUDA-capable GPU (Recommended for faster inference)
* Internet connection (to download model weights)

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Deckku/CardioECHO.git
   cd CardioECHO
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *OR run:*
   ```bash
   pip install numpy opencv-python torch torchvision pydicom reportlab matplotlib PyQt6
   ```

3. **Download Model Weights (Required):**
   
   The application requires three pre-trained model files (~600 MB total).
   
   1. Download the files from this link: **[Google Drive Link](https://drive.google.com/drive/folders/17QpRCyoG1jDRxe4NjTJuRY2KwkoM5RPy?usp=sharing)**
   2. Place the downloaded files in the **root directory** of this project (the same folder as `main_app_final.py`)
   3. Ensure the filenames are exactly as follows:
      * `segmentation.pt`
      * `esv.pth`
      * `ef.pt`

---

## Usage

### Running the Python Script

Once the dependencies are installed and models are placed in the root folder, run:

```bash
python main_app_final.py
```

### Building the Executable (Optional)

If you wish to create a standalone Windows executable (.exe) so the application can run without installing Python, follow these steps:

1. **Install PyInstaller:**
   ```bash
   pip install pyinstaller
   ```

2. **Run the Build Command:**
   
   Use the following command to package the application. This command ensures that the large model files and the cv2 dependency are correctly bundled inside the application folder.
   
   ```powershell
   python -m PyInstaller --noconsole --onedir --name="CardioECHO" --clean --collect-all cv2 --hidden-import=modulefinder --add-data "segmentation.pt;." --add-data "esv.pth;." --add-data "ef.pt;." main_app_final.py
   ```

3. **Locate the App:**
   * Go to the newly created `dist/` folder
   * Open the `CardioECHO` folder
   * Run `CardioECHO.exe`

> **Note:** When moving the application to another computer, you must copy the entire `CardioECHO` folder, not just the executable file.

---

## Project Structure

* **main_app_final.py:** The entry point for the GUI application and analysis logic
* **Report.pdf:** Comprehensive project documentation and feasibility analysis
* **requirements.txt:** List of Python library dependencies
* **.gitignore:** Configuration to exclude build artifacts and large model files from the repository

---

## Credits

**Author:** Azami Hassani Adnane  
Student, ENSAM Meknès (GIIADS)

**Supervisor:** M. Hosni Mohamed  
Professor, ENSAM Meknès

---

## Dataset & Acknowledgments

### EchoNet-Dynamic Dataset

This project relies on the EchoNet-Dynamic dataset, a large-scale database of cardiac echocardiograms provided by Stanford University. We gratefully acknowledge the researchers and the institution for making this data publicly available for scientific research.

**Reference:**  
Ouyang, D., He, B., Ghorbani, A., Yuan, N., Ebinger, J., Langlotz, C. P., ... & Zou, J. Y. (2020). Video-based AI for beat-to-beat assessment of cardiac function. *Nature, 580*(7802), 252-256.
