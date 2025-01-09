# PointPillars OpenVINO™ Demo on Intel GPU

## Introduction

This repository provides five key demonstrations:

- The implementation is adapted from the [Demo of PointPillars Optimization](https://github.com/intel/OpenVINO-optimization-for-PointPillars), showcasing how to implement and optimize PointPillars on Intel platforms using OpenVINO™. The original code sources are [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), which sets up the PointPillars pipeline demo, and [SmallMunich](https://github.com/SmallMunich/nutonomy_pointpillars), which converts the PointPillars PyTorch model to ONNX format. For more technical details, refer to the [Optimization of PointPillars by using Intel® Distribution of OpenVINO™ Toolkit](https://www.intel.com/content/www/us/en/developer/articles/technical/optimization-of-pointpillars.html).

- The repository supports Intel MTL iGPU and Arc770 dGPU platforms.
- It adds an INT8 quantization method for RPN and PFE models.
- Scatter latency statistics are included in the demo outputs.
- Validation has been conducted on Intel MTL iGPU and Arc770 dGPU devices.

---

## Overview

This document provides detailed instructions for setting up and running the PointPillars OpenVINO™ Demo on Intel GPU. The sections below include hardware and software requirements, demo setup steps, and quantization guidance.

### Sections:

1. [Requirements](#requirements)
   - [Hardware](#hardware)
   - [Software](#software)
2. [Demo Setup](#demo-setup)
   - [Install iGPU or dGPU Drivers](#1-install-igpu-or-dgpu-drivers)
   - [Install Development and Compilation Tools](#2-install-development-and-compilation-tools)
   - [Clone the Repository and Create an Environment](#3-clone-the-repository-and-create-an-environment)
   - [Install Python Packages](#4-install-python-packages)
   - [Prepare Datasets](#5-prepare-datasets)
   - [Set Up OpenPCDet Package](#6-set-up-openpcdet-package)
   - [Run the Demo](#7-run-the-demo)
3. [Quantization](#quantization)
   - [Prerequisites](#1-prerequisites)
   - [Update Dataset Path](#2-update-dataset-path)
   - [Quantize the Model](#3-quantize-the-model)

---

## Requirements

### Hardware

Choose one of the following hardware setups:
- Intel MTL with iGPU
- Intel Arc770 dGPU + Core CPU

### Software

- Ubuntu 22.04
- Linux Kernel 6.5.0-18-generic
- Python 3.10
- OpenVINO™ 2024.3

---

## Demo Setup

### 1. Install iGPU or dGPU Drivers

#### 1.1 MTL iGPU Driver Installation

Refer to the [compute-runtime releases](https://github.com/intel/compute-runtime/releases/tag/24.26.30049.6).

#### 1.2 Arc770 Driver Installation

Refer to the [Intel Arc GPU documentation](https://dgpu-docs.intel.com/driver/client/overview.html#installing-client-gpus-on-ubuntu-desktop-22-04-lts).

---

### 2. Install Development and Compilation Tools

```bash
sudo apt update
sudo apt-get install python3-dev
sudo apt install build-essential
```

---

### 3. Clone the Repository and Create an Environment

```bash
cd /home/shawn
mkdir project
git clone https://github.com/shawn9977/PointPillars-Demo.git
cd PointPillars-Demo
python3 -m venv env_PointPillars
source env_PointPillars/bin/activate
```

---

### 4. Install Python Packages

```bash
pip install openvino-dev nncf
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

cd /home/shawn/project/PointPillars-Demo
pip install -r requirements.txt
```

---

### 5. Prepare Datasets

Refer to [SmallMunich's repo](https://github.com/SmallMunich/nutonomy_pointpillars) for dataset generation.

**Note:** This step requires an NVIDIA GPU environment. Alternatively, you can skip this step as pre-generated datasets are included in this repository under `PointPillars-Demo/main/datasets/training/velodyne_reduced/`.

Set the dataset path environment variable:

```bash
export my_dataset_path=<your_dataset_folder>/training/velodyne_reduced

# Example:
export my_dataset_path=/home/shawn/project/PointPillars-Demo/datasets/training/velodyne_reduced
```

---

### 6. Set Up OpenPCDet Package

```bash
python setup.py develop
```

---

### 7. Run the Demo

```bash
cd /home/shawn/project/PointPillars-Demo/tools/
python demo.py --cfg_file pointpillar.yaml --num -1 --data_path $my_dataset_path

# If `my_dataset_path` is not set, run:
python demo.py --cfg_file pointpillar.yaml --num 100 --data_path /home/shawn/project/PointPillars-Demo/datasets/training/velodyne_reduced
```

Demo outputs include performance metrics:

```bash
INFO  -----------------Quick Demo of OpenPCDet-------------------------
INFO  Loading the dataset and model.
INFO  Number of samples in dataset: xxx
INFO  ------Run number of samples: xxx in mode: balance
INFO  Total: xxx seconds
INFO  FPS: xxx
INFO  Latency: xxx milliseconds
INFO  Scatter latency: xxx milliseconds

INFO  Demo done.
```

---

## Quantization

### 1. Prerequisites

Complete steps 1-6 from the demo setup.

---

### 2. Update Dataset Path

Edit `quant.py` to set the dataset path:

```python
DATA_PATH = "/home/shawn/project/PointPillars-Demo/datasets/training/velodyne_reduced"
```

---

### 3. Quantize the Model

Run `quant.py` to save the quantized model (`quantized_pfe.xml`) in the `tools` directory:

```bash
cd /home/shawn/project/PointPillars-Demo/tools/
python quant.py
