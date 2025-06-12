# 🐎 Horses Or Humans 🧍

A full ML pipeline to classify images of horses and humans using state-of-the-art convolutional neural networks. This project includes everything from data preprocessing and training to model evaluation, inference, and a live web interface using Gradio and Docker.

---

## 🔧 Tech Stack

- **Frameworks:** PyTorch, Hydra, MLflow
- **Image Processing:** PIL, OpenCV
- **Model Architectures:** EfficientNet-B0, ConvNeXt, ResNet-18
- **Interface:** Gradio
- **Environment Management:** Miniconda
- **Containerization:** Docker Desktop

---

## 🧠 Project Features

- ✅ Complete ML pipeline from raw image data to deployment
- 🎯 High-performance classification with multiple CNN architectures
- 🔍 Grad-CAM visualization for model explainability
- 📈 MLflow for logging experiments and hyperparameters
- 🐳 Docker support for both CPU and GPU environments
- 💡 Interactive Gradio UI for real-time image classification

---

## 📂 Directory Overview

/
Horses or Humans
├── conf
│ ├── data_processing.yaml
│ ├── hyperparameter.yaml
│ ├── inference.yaml
│ ├── logging.yaml
│ ├── training.yaml
│ └── environ
│ └── default.yaml
├── data
│ ├── processed
│ │ ├── train
│ │ ├── test
│ │ └── val
│ ├── unprocessed
│ │ ├── horses
│ │ └── humans
│ ├── diagrams
│ │ └── Horses or Humans.drawio
│ ├── docker
│ │ ├── horsesorhumans-cpu.Dockerfile
│ │ └── horsesorhumans-gpu.Dockerfile
│ ├── images
│ ├── logs
│ ├── mlflow
│ │ └── optuna_hparam_tuning.db
│ ├── model
│ ├── notebook
│ │ └── hyperparameter_tuning_analysis.ipynb
│ ├── scripts
│ │ ├── build_cpu_docker.sh
│ │ ├── build_gpu_docker.sh
│ │ └── run_docker.sh
│ ├── slides
│ │ └── Horses or Humans.pptx
│ └── src
│ ├── app.py
│ ├── hyerparameter.py
│ ├── infer.py
│ ├── process_data.py
│ ├── train.py
│ ├── data_prep
│ │ └── pipelines.py.py
│ ├── hyperparameter_tuning
│ │ └── hyperparameter_tuning.py
│ ├── inference
│ │ └── inference_pipeline.py
│ ├── training_pipelines
│ │ └── training_pipelines.py
│ └── utils
│ ├── general_utils.py
│ └── seed_utils.py
├── .dockerignore
├── dev-requirements.txt
├── horsesorhumans-conda-env.yaml
├── requirements-cpu.txt
└── requirements-gpu.txt

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/horses-or-humans.git
cd horses-or-humans
```

### 2. Set Up Environment

conda env create -f horsesorhumans-conda-env.yaml
conda activate horsesorhumans

### 3. Run Training

python src/train.py

### 4. Launch Gradio App

python src/app.py

## 🐳 Docker Support

### CPU Build

bash data/scripts/build_cpu_docker.sh

### GPU Build

bash data/scripts/build_gpu_docker.sh

### Run Docker

bash data/scripts/run_docker.sh

## 📊 Model Performance

Model Accuracy (Val) F1 Score Params
EfficientNet-B0 92.8% 0.927 ~5.3M
ConvNeXt-T 91.2% 0.914 ~29M
ResNet-18 88.5% 0.886 ~11.7M
