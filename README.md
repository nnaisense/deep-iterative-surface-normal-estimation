## Deep Iterative Surface Normal Estimation
Code repository for the paper [<i>Deep Iterative Surface Normal Estimation</i>](https://arxiv.org/abs/1904.07172), CVPR 2020 (accepted for publication), by Jan Eric Lenssen, Christian Osendorfer and Jonathan Masci @NNAISENSE.
<p align="center">
  <img width="40%" src="overview.png?sanitize=true"/>
</p>

Below, we explain how to
* install the code,
* reproduce paper results for the PCPNet and NYU datasets,
* train a new model

Further, we provide a short overview of important classes and functions.

### Dependencies
The code runs with Python 3.7, CUDA 10.0. The following additional dependencies need to be installed:
* Numpy
* Scipy
* **[PyTorch](https://pytorch.org/)**
* **[PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)**
* **[torch_sym3eig](https://github.com/nnaisense/pytorch_sym3eig)**

### Install
After installing dependencies, clone repository and run setup.py to compile the quaternion GPU kernels:

```
git clone https://github.com/nnaisense/deep-iterative-surface-normal-estimation
cd deep-iterative-surface-normal-estimation
python setup.py install
```

### Run Normal Estimation
To reproduce evaluation on the PCPNet test dataset, run normals_pcpnetdata_eval.py with the following parameters:
* --model_name: Model file from trained_models/ to use \[default: 'network_k64.pt'\] 
* --dataset_path: Path to store PCPNet dataset (is downloaded automatically) \[default: 'data/pcpnet_data/'\] 
* --k_test: Neighborhood size for testing \[default: 64\]
* --iterations: Number of iterations for testing \[default: 4\]
* --results_path: If set, resulting normals are stored in the given folder \[default: None\]

Example:

```
python normals_pcpnetdata_eval.py --model_name='network_k64.pt' --k_test=64 --iterations=4
```

We provide models trained on the PCPNet train dataset with k=32,48,64,96,128 in 'trained_models/'.

To reproduce evaluation on the NYU Depth V2 dataset run normals_nyudepthv2_eval.py with the following parameters:
* --model_name: Model file from trained_models/ to use \[default: 'network_k64.pt'\] 
* --dataset_path: Path to store NYU Depth V2 dataset (is downloaded automatically) \[default: 'data/nyudepthv2/'\] 
* --k_test: Neighborhood size for testing \[default: 64\]
* --iterations: Number of iterations for testing \[default: 4\]
* --results_path: Folder in which resulting images are stored \[default: 'nyu_out/'\]

Example:

```
python normals_nyudepthv2_eval.py --model_name='network_k64.pt' --k_test=64 --iterations=4
```

### Training
To train on the PCPNet train dataset, run normals_pcpnetdata_train.py with the following parameters:
* --model_name: Model file name to store in trained_models/. Needs epoch placeholder {}.  \[default: 'network_new_epoch{}.pt'\] 
* --dataset_path: Path to store PCPNet dataset (is downloaded automatically) \[default: 'data/pcpnet_data/'\] 
* --k_train: Neighborhood size for training \[default: 48\]
* --iterations: Number of iterations for training \[default: 8\]

Example:

```
python normals_pcpnetdata_train.py --model_name='network_k48new_epoch{}.pt' --k_train=48 --iterations=8
```

Remark: Currently, training with larger k requires a large amount of GPU memory 

### Functionality
```python
class NormalEstimation(torch.nn.Module)
```
Provides the main algorithm for normal estimation. The forward() function computes one iteration of the algorithm, including GNN re-weighting and least squares fitting.


```python
eig_val, eig_vec = Sym3Eig.apply(cov)
```
Performs the least squares fitting through parallel (batch-wise) 3x3 eigendecomposition as provided by our **[torch_sym3eig](https://github.com/nnaisense/pytorch_sym3eig)** module.


```python
class GNNFixedK(torch.nn.Module)
class GNNVariableK(torch.nn.Module)
```
Define the GNNs for re-weighting in network/gnn.py. The version for fixed neighborhood size k is slighty faster due to dim-reduce instead of scatter operations. Except that, they are similar and can be used interchangeably, depending on use case.