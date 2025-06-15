
# Brain-Computer Interface (BCI) for Motor Imagery Classification

This repository contains Python scripts for a Brain-Computer Interface (BCI) system that classifies motor imagery EEG signals using deep learning models, particularly ATC-Net. The system can predict movement intentions and control virtual/robotic arms accordingly.

## 📋 Contents
1. [Requirements](#-requirements)
2. [Installation](#-installation)
3. [Project Structure](#-project-structure)
4. [Scripts Overview](#-scripts-overview)
5. [Usage](#-usage)
6. [License](#-license)

## 📦 Requirements
- Python 3.8+
- Essential packages:
  ```
  pip install tensorflow numpy scipy matplotlib mne pylsl colorama
  ```
- For GPU acceleration: CUDA-enabled TensorFlow
- Emotiv EPOC X headset (or compatible EEG device)

## 🚀 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bci-motor-imagery.git
   cd bci-motor-imagery
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Project Structure
```
.
├── main.py                # Main execution script for real-time BCI control
├── MainValKeras.py        # Model training and evaluation script
├── ModelKeras.py          # ATC-Net and other model architectures
├── preprocessingkeras.py  # EEG data preprocessing utilities
├── attentionModels.py     # Attention mechanisms for models
├── headset/               # EEG headset interface code
│   ├── emotiv_lsl/        # Emotiv LSL server implementation
│   ├── run.py             # Headset initialization
│   └── record.py          # EEG data recording
├── vrepcom/               # CoppeliaSim integration
└── examples/              # Example scripts
```

## 🛠 Scripts Overview

### 1. Main Execution (`main.py`)
Controls the virtual arm using predicted motor imagery:
```bash
python main.py
```
- Uses pre-trained ATC-Net model
- Records EEG data in real-time
- Predicts movement intention (left/right/up/down)
- Controls CoppeliaSim virtual arm

### 2. Model Training (`MainValKeras.py`)
Trains and evaluates ATC-Net models:
```bash
python MainValKeras.py
```
- Supports multiple datasets (BCI2a, HGD, CS2R)
- Implements leave-one-subject-out (LOSO) evaluation
- Generates performance metrics and visualizations

### 3. EEG Recording (`record.py`)
Records and preprocesses EEG data:
```python
from headset.record import record
eeg_data = record(SRATE=128, duration=9)  # Returns (14, 1125) array
```

### 4. Headset Interface (`run.py`)
Initializes Emotiv EPOC X headset:
```python
from headset.run import start_device
start_device(SRATE=128)
```

## 🏃‍♂️ Usage

### Real-time Control
1. Start the LSL server:
   ```bash
   python -m headset.emotiv_lsl.emotiv_epoc_x
   ```
2. Run the main BCI control script:
   ```bash
   python main.py
   ```

### Model Training
1. Prepare dataset in `BCI2a/` directory
2. Train model:
   ```bash
   python MainValKeras.py
   ```
3. Results saved in `Results/`

### Data Collection
```python
from headset.record import record
eeg_data = record()  # 9-second recording
```

## 📜 License
Apache License 2.0

## 📚 References
- ATC-Net paper: [IEEE TII 2022](https://doi.org/10.1109/TII.2022.3197419)
- BCI Competition IV-2a dataset
- Emotiv EPOC X documentation
```

