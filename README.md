# PT-Brain

Physics-Transfer (PT) theory integrates physical constraints into learning frameworks. Using this theory, we predict brain morphogenesis through a physics-grounded transfer framework that leverages a chain of anchor domains, ensuring the generalization bound of our inference. Specifically, models are trained exclusively on the source domain (sphere) and then directly applied to predict brain morphology in the target domain.

This README is organized into four main sections:

1. **Installation** – Step-by-step instructions for setting up the environment.
2. **Data** – Guidance for downloading the training data (sphere data) and the inference data (brain data).
3. **Training** – Example notebooks demonstrating how to train statistical learning (SL) and PT models on the sphere data.
4. **Inference & Analysis** – Example notebooks showing how to predict brain data using the trained SL and PT models.

---

## 1. Installation

This project depends on the following deep learning frameworks:
- PyTorch 1.13.1 + CUDA 11.7
- PyTorch Geometric (PyG)
- PyTorch3D
- Python 3.9
Different CUDA versions correspond to different pre-built PyTorch, PyG, and PyTorch3D packages. Make sure your system has CUDA 11.7 installed. If your CUDA version is different, replace the wheels with the corresponding version; otherwise, installation may fail.

### 1.1 Create and Activate Conda Environment
```bash
conda create -n pt-brain python=3.9
conda activate pt-brain
```

### 1.2 Install PyTorch
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --index-url https://download.pytorch.org/whl/cu117
```

### 1.3 Install PyTorch Geometric (PyG)
```bash
pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-sparse==0.6.15 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-geometric==2.3.1
```

### 1.4 Install Jupyter
```bash
pip install jupyter
```

### 1.5 Install PyTorch3D
```bash
pip install pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt1131/download.html
```

---

## 2. Data

The dataset can be downloaded from the Figshare link: [https://doi.org/10.6084/m9.figshare.30812246](https://doi.org/10.6084/m9.figshare.30812246).

**Folder structure:**

`Train_Data/`
- Contains sphere data used for training the SL and PT models.

`Inference_Data/`
- Contains ellipsoid and brain data used for testing the SL and PT models.

**Model files:**

- `SL.pth` — Trained SL model.

- `PT.pth` — Trained PT model.

---

## 3. Training

After downloading the data, you can run the code cells in the Jupyter Notebook files (`SL_Training.ipynb` and `PT_Training.ipynb`) step by step to train the SL and PT models. Once the training is complete, the trained model files (`SL.pth` and `PT.pth`) will be generated.

## 4. Inference & Analysis
After training the models (you can also directly use our well-trained models (`SL.pth` and `PT.pth`)), you can run the code cells in the Jupyter Notebook files (`SL_Inference_Analysis.ipynb` and `PT_Inference_Analysis.ipynb`) step by step to evaluate the SL and PT models on the ellipsoid and brain data, as well as to analyze the neural activation patterns.


