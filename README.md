# Bamboleo Game Object Detection

This project implements two approaches to detect objects in the Bamboleo game. The goal is to accurately identify and track objects within the game environment.

## Approaches

### Approach 1: [SAM2 Approach]
- **Description**: Implementation using SAM2 to track objects.

### Approach 2: [Color-based Approach]
- **Description**: Implementation using color information and other built-in functions from OpenCV.

## Installation
1. Install torch>2.5.1:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
2. Install SAM2:
   ```bash
   pip install 'git+https://github.com/facebookresearch/sam2.git'
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the fisrt approach:
```bash
python sam.py --video_input <video_input.mp4>
```

To run the second approach:
```bash
python color_based.py --video_input <video_input.mp4>
```
