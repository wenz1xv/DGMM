# DGMM: Deep Genetic Molecular modification Model

## Description
This repository provides the official implementation of the paper titled "DGMM: A Deep Learning-Genetic Algorithm Framework for Ef-ficient Lead Optimization in Drug Discovery" The DGMM is a powerful model designed to enable advanced molecular modification using genetic algorithm principles, pushing the boundaries of molecular design and optimization.

## Install (Linux)

### Dependence
- conda 4.12
- python 3.12
- tensorflow 2.19.0
- autodock vina/ Unidock / Schrödinger Glide
- rdkit 2023.9.1

### Online Install
> Ensure that `conda` and `git` are pre-installed on your system.
```bash
git clone https://github.com/wenz1xv/DGMM.git
conda create -n dgmm -python=3.12 -c conda-forge
conda activate dgmm
# since newest unidock changed, we use v. 1.1.2
conda install -c conda-forge unidock==1.1.2
pip install -r requirement.txt
```

> you can either register conda env to jupyter or install jupyer in dgmm conda env

```bash
# register dgmm to jupyer existed

conda activate dgmm
python -m ipykernel install --user --name dgmm --display-name "dgmm"

```


## Usage (Linux)

> If you encounter warnings like `Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file` or you see messages related to `cudart stub`, __do not worry__. These can be safely ignored if you do not have a GPU set up on your machine. The code will still work on **CPU mode**.

### Training the Model


Additional training data can be downloaded:

- Download `zinc_train_ext.h5` from [Google Drive](https://drive.google.com/drive/folders/1oB63AxrvwGbI8GmFR-5eE3xFwK61QmCs?usp=drive_link).

Follow the detailed instructions within the Jupyter notebooks:

- Use `VAE_train.ipynb` for training the Variational Autoencoder (VAE) module.

### Running the DGMM
Refer to `DGMM_[SCH/UniDocl/Vina].ipynb` for detailed guidance on running the DGMM model.

> Note: Ensure that Schrödinger is installed on your system if you intend to use **Glide Docking**.

## Citation
If you use DGMM in your research or incorporate our algorithms in your work, please cite our paper:

**DGMM: A Deep Learning-Genetic Algorithm Framework for Efficient Lead Optimization in Drug Discovery**
Jiebin Fang, Churu Mao, Yuchen Zhu, Xiaoming Chen, Yun Huang, Wanjing Ding, Chang-Yu Hsieh, and Zhongjun Ma
*Journal of Chemical Information and Modeling*, 2025, 65(15), 8168–8180.
https://doi.org/10.1021/acs.jcim.5c01017

```bibtex
@article{fang2025dgmm,
  title = {DGMM: A Deep Learning-Genetic Algorithm Framework for Efficient Lead Optimization in Drug Discovery},
  author = {Fang, Jiebin and Mao, Churu and Zhu, Yuchen and Chen, Xiaoming and Huang, Yun and Ding, Wanjing and Hsieh, Chang-Yu and Ma, Zhongjun},
  journal = {Journal of Chemical Information and Modeling},
  year = {2025},
  volume = {65},
  number = {15},
  pages = {8168--8180},
  doi = {10.1021/acs.jcim.5c01017}
}
```
