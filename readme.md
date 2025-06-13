# 🐎 Horses Or Humans 🧍

[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/hydra-0D86FF?logo=hydra&logoColor=white)](https://hydra.cc/)
[![MLflow](https://img.shields.io/badge/mlflow-13B9FD?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A full ML pipeline to classify images of horses and humans using state-of-the-art convolutional neural networks.  
This project covers data preprocessing, training, hyperparameter tuning, inference with Grad-CAM explainability, and a live web UI using Gradio and Docker.

---

## 🔧 Tech Stack

- **Frameworks:** PyTorch, Hydra, MLflow
- **Image Processing:** PIL, OpenCV
- **Model Architectures:** EfficientNet-B0, ConvNeXt-Tiny, ResNet-18
- **Interface:** Gradio
- **Environment:** Miniconda
- **Containerization:** Docker (CPU & GPU support)

---

## 🧠 Project Features

- ✅ End-to-end ML pipeline from raw images to deployment
- 🎯 Multiple CNN backbones with competitive accuracy
- 🔍 Grad-CAM for visual explanations of predictions
- 📈 MLflow experiment tracking & hyperparameter logging
- 🐳 Dockerized for easy local and cloud deployment (CPU/GPU)
- 💡 Interactive Gradio app for real-time classification

---

## 📂 Directory Overview

<pre> ```plaintext /
Horses or Humans
├── conf/                      # Config files for training, inference, logging, env
├── data/
│   ├── processed/             # Processed datasets (train, val, test)
│   ├── unprocessed/           # Raw images (horses, humans)
│   ├── diagrams/              # Project diagrams (drawio)
│   ├── docker/                # Dockerfiles for CPU & GPU
│   ├── images/                # Extra images
│   ├── logs/                  # Logs for training and inference
│   ├── mlflow/                # MLflow DB for experiment tracking
│   ├── model/                 # Saved models & checkpoints
│   ├── notebook/              # Jupyter notebooks for analysis
│   ├── scripts/               # Shell scripts for docker build/run
│   ├── slides/                # Presentation slides
│   └── src/                   # Source code
│       ├── app.py             # Gradio frontend
│       ├── train.py           # Training entrypoint
│       ├── infer.py           # Inference entrypoint
│       ├── process_data.py    # Data processing entrypoint
│       ├── hyperparameter.py  # Hyperparameter tuning entrypoint
│       ├── data_prep/         
│       ├── hyperparameter_tuning/
│       ├── inference/
│       ├── training_pipelines/
│       └── utils/
├── .dockerignore
├── dev-requirements.txt
├── horsesorhumans-conda-env.yaml
├── requirements-cpu.txt
└── requirements-gpu.txt
``` </pre>

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/lydra4/Deep-Learning/tree/main/Horses%20or%20Humans
cd horses-or-humans
```

### 2. Set Up Environment

```bash
conda env create -f horsesorhumans-conda-env.yaml
conda activate horsesorhumans
```

### 3. Prepare Dataset

```bash
python src/process_data.py
```

### 4. Train Model

```bash
python src/train.py
```

### 5. Launch Gradio App

```bash
python src/app.py
```

## 🐳 Docker Support

### CPU Build

```
bash data/scripts/build_cpu_docker.sh
```

### GPU Build

```bash
bash data/scripts/build_gpu_docker.sh
```

### Run Docker

```bash
bash data/scripts/run_docker.sh
```

## 📊 Model Performance

📊 Model Performance (Validation Set)

| Model           | Train Accuracy | Val Accuracy | Parameters |
| --------------- | -------------- | ------------ | ---------- |
| EfficientNet-B0 | 92.8%          | 0.927        | ~5.3M      |
| ConvNeXt-T      | 91.2%          | 0.914        | ~29M       |
| ResNet-18       | 88.5%          | 0.886        | ~11.5M     |

## 📚 References

[![PyTorch](https://img.shields.io/badge/pytorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/hydra-0D86FF?logo=hydra&logoColor=white)](https://hydra.cc/)
[![MLflow](https://img.shields.io/badge/mlflow-13B9FD?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Gradio](https://img.shields.io/badge/gradio-6B40E3?logo=gradio&logoColor=white)](https://gradio.app/)
[![Optuna](https://img.shields.io/badge/optuna-4F3BFF?logo=optuna&logoColor=white)](https://optuna.org/)
