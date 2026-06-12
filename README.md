
# Stereo Vision for Retinal Microsurgical Tool Tracking

## Project Overview
This project focuses on depth perception in retinal microsurgery. Participants will explore how stereo video feeds from surgical microscopes can be processed to extract depth maps, enabling computational analysis for robotic and computer-assisted interventions.

## Project Roadmap
1. **Data Collection**: Record stereoscopic videos of phantom retinal procedures using Media Express.
2. **Calibration**: Use a checkerboard pattern to compute intrinsic and extrinsic camera matrices.
3. **Rectification**: Align the Left and Right feeds to epipolar lines.
4. **Depth Estimation**: Generate depth maps using Semi-Global Block Matching (SGBM).
5. **Analysis**: Evaluate depth accuracy and visualize surgical tool movement.

## Pre-Workshop Workbook
### 1. Software & Environment Setup
Please ensure you have a working development environment on your machine on Monday.
- Python: Install Python 3.10 or higher.
- VS Code: Install VS Code.
- Clone the Project Repo: [Retinal Microsurgery Project Repo](https://github.com/AdrianaNam/retinal-microsurgery-project.git)

### 2. Core Concepts (Recommended Reading)
To hit the ground running, please review these key concepts:

Stereo Vision Basics: Understand the concept of "Epipolar Geometry."
- Stereo Vision and Depth Estimation Tutorial
Camera Calibration: Why do we calibrate?
- Camera Calibration with OpenCV
Surgical Context: Learn how depth perception aids microsurgeons.
- Search Term: "Computer-assisted retinal surgery depth perception"

### 3. Tutorial Material for the Workshop

We will be using these during the sessions. Feel free to browse them:
-  OpenCV Depth Maps: Stereo Matching with SGBM
- Rectification: Learn how to align images before calculating disparity.

### Workshop Schedule
- Day 1: Understanding the hardware (Media Express capture) and data extraction.
- Days 1 & 2 (Afternoon): Camera calibration using the lab's grid and data collection.
- Day 3: Applying Semi-Global Block Matching (SGBM) to create depth maps.
- Days 4 & 5: Finalizing the pipeline, analysing results, and preparing your final presentation.


## Getting Started
### 1. Environment Setup
- Install [Python 3.10+](https://www.python.org/) and [VS Code](https://code.visualstudio.com/).
- Clone this repository: `git clone https://github.com/AdrianaNam/retinal-microsurgery-project.git`
- Install dependencies:
  ```bash
  pip install numpy opencv-python torch torchvision pyyaml


### 2. Data
- Use Media Express (configured to 1080i50 3D) to record your surgical trials.
- Save files as left.avi and right.avi.
- Ensure tool movement is captured from multiple angles.

### 3. Script skeletons provided
- scripts/calibrate.py: Input your calibration images to generate calib_params.yaml.
- scripts/depth_estimation.py: Uses calib_params.yaml to process frames.

### 4. Annotation Tool
If you need to label surgical tools for tracking or validation, use the following annotation tool:

- [Segment Anything Annotator v2](https://github.com/RViMLab/segment-anything-annotator-v2)

## Workflow

- Calibration: Use the provided scripts/calibrate.py with the calibration grid provided in the lab.

- Depth Mapping: Run scripts/depth_estimation.py to generate disparity maps.

- Analysis: Compare estimated tool depth against ground truth.

## Collaboration

- Please work on your assigned branches.
- Use git pull frequently to sync changes.
- Submit your final results and code via Pull Requests.