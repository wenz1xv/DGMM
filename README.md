# DGMM: Deep Genetic Molecular modification Model

## Description
This repository provides the official implementation of the paper titled "Human-level Molecular Optimization Driven by Mol-Gene Evolution. " The DGMM is a powerful model designed to enable advanced molecular modification using genetic algorithm principles, pushing the boundaries of molecular design and optimization.
## Install (Linux)

### Dependence
- conda 4.12
- python 3.7
- tensorflow 2.9.1
- autodock vina
- rdkit

### Online Install
> Install conda and git first
```
git clone https://github.com/wenz1xv/DGMM.git
conda create -n dgmm -python=3.7
conda activate dgmm
pip install -r requirement.txt
```

### Online Installation
> Ensure that `conda` and `git` are pre-installed on your system.
```
git clone https://github.com/wenz1xv/GDMM.git
conda create -n dgmm python=3.7
conda activate dgmm
pip install -r requirement.txt
```

### Offline Installation
> Make sure `conda` is installed on your system first.

1. Download the file `py37.tar.gz` from [Google Drive](https://drive.google.com/drive/folders/1oB63AxrvwGbI8GmFR-5eE3xFwK61QmCs?usp=drive_link) to your `your_conda_path`.
2. Create the environment directory:
   ```
   mkdir your_conda_path/envs/dgmm
   ```
3. Move the downloaded `py37.tar.gz` to your environment path:
   ```
   mv py37.tar.gz your_conda_path/envs/dgmm/
   ```
4. Navigate to the environment directory:
   ```
   cd your_conda_path/envs/dgmm
   ```
5. Extract the environment files:
   ```
   tar xvzf py37.tar.gz
   ```
6. Activate the environment:
   ```
   conda activate dgmm
   ```

## Usage (Linux)

> If you encounter warnings like `Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file` or you see messages related to `cudart stub`, __do not worry__. These can be safely ignored if you do not have a GPU set up on your machine. The code will still work on **CPU mode**.

### Training the Model


Additional training data can be downloaded:

- Download `zinc_train_ext.h5` from [Google Drive](https://drive.google.com/drive/folders/1oB63AxrvwGbI8GmFR-5eE3xFwK61QmCs?usp=drive_link).

Follow the detailed instructions within the Jupyter notebooks:

- Use `TVAE_train.ipynb` for training the Teacher Variational Autoencoder (TVAE) module.
- Use `DVAE_train.ipynb` for training the Discrete Variational Autoencoder (DVAE) module.

### Running the DGMM
Refer to `DGMM_run.ipynb` for detailed guidance on running the DGMM model.

> Note: Ensure that Schrödinger is installed on your system if you intend to use **Glide Docking**.

## Citation
If you use DGMM in your research or incorporate our algorithms in your work, please cite our paper:

*(Provide citation information here once available.)*