# [Re] Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL
## A replication of Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL by Strodthoff et al. 2021

This github repository comprises our code replicating the experiments reported in *Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL*

Full reference to the original paper :
> N. Strodthoff, P. Wagner, T. Schaeffter, and W. Samek, ‘Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL’, IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 5, pp. 1519–1528, May 2021, doi: 10.1109/JBHI.2020.3022989.


Original github repository: https://github.com/helme/ecg_ptbxl_benchmarking

The main goal of this project was to reproduce the results from Strodthoff et al (2021). In addition, we tested the robustness of the proposed models by adding random noise to the ECGs in the test set. Finally, we  used the provided template to implement a new model and evaluated it on the six benchmark tasks described in Strodthoff et al.


## Setup and requirements

To re-run our replication experiments, simply upload the notebook `PTB_XL_experiments.ipynb` to [Google colab](https://colab.research.google.com/) and run the code cells. In the second code cell you will be asked to mount your Google Drive to the Google colab notebook. This is not mandatory, but it is recommended if you want to store the results the experiments.

### Data
The dataset (PTB-XL) will be downloaded from the [original data repository](https://physionet.org/content/ptb-xl/1.0.3/)  in the 5th code cell of the `PTB_XL_experiments.ipynb` notebook.

### Dependencies
A [custom version of Fast AI](https://github.com/Bsingstad/fastai) was created to make the original repository compatible with Google Colab notebooks. This is taken care of in code cell 11 in `PTB_XL_experiments.ipynb`.

## Results

We encourage other authors to share their results on this dataset by submitting a PR. The evaluation proceeds as described in the manuscripts:
The reported scores are test set scores (fold 10) as output of the above evaluation procedure and should **not be used for hyperparameter tuning or model selection**. In the provided code, we use folds 1-8 for training, fold 9 as validation set and fold 10 as test set. We encourage to submit also the prediction results (`preds`, `targs`, `classes` saved as numpy arrays `preds_x.npy` and `targs_x.npy` and `classes_x.npy`) to ensure full reproducibility and to make source code and/or pretrained models available.

 #### 1. PTB-XL: all statements

| Model | AUC &darr; | paper/source | code |
|---:|:---|:---|:---|
| **Inception Time** | **0.926(08)** | **our work** | **this repo** |
| inception1d | 0.925(08) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| xresnet1d101 | 0.925(07) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| resnet1d_wang | 0.919(08) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| fcn_wang | 0.918(08) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| lstm_bidir | 0.914(08) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| lstm | 0.907(08) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| Wavelet+NN | 0.849(13) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|

 #### 2. PTB-XL: diagnostic statements

| Model | AUC &darr; | paper/source | code |
|---:|:---|:---|:---|
| xresnet1d101 | 0.937(08) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| resnet1d_wang | 0.936(08) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| lstm_bidir | 0.932(07) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| inception1d | 0.931(09) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| **Inception Time** | **0.929(09)** | **our work** | **this repo** |
| lstm | 0.927(08) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| fcn_wang | 0.926(10) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| Wavelet+NN | 0.855(15) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|

 #### 3. PTB-XL: Diagnostic subclasses

| Model | AUC &darr; | paper/source | code |
|---:|:---|:---|:---|
| inception1d | 0.930(10) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| xresnet1d101 | 0.929(14) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| lstm | 0.928(10) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| resnet1d_wang | 0.928(10) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| fcn_wang | 0.927(11) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| **Inception Time** | **0.927(08)** | **our work** | **this repo** |
| lstm_bidir | 0.923(12) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| Wavelet+NN | 0.859(16) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|

 #### 4. PTB-XL: Diagnostic superclasses

| Model | AUC &darr; | paper/source | code |
|---:|:---|:---|:---|
| resnet1d_wang | 0.930(05) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| xresnet1d101 | 0.928(05) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| lstm | 0.927(05) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| fcn_wang | 0.925(06) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| **Inception Time** | **0.922(06)** | **our work** | **this repo** |
| inception1d | 0.921(06) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| lstm_bidir | 0.921(06) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| Wavelet+NN | 0.874(07) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|

 #### 5. PTB-XL: Form statements

| Model | AUC &darr; | paper/source | code |
|---:|:---|:---|:---|
| inception1d | 0.899(22) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| xresnet1d101 | 0.896(12) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| resnet1d_wang | 0.880(15) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| lstm_bidir | 0.876(15) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| fcn_wang | 0.869(12) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| lstm | 0.851(15) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| **Inception Time** | **0.840(11)** | **our work** | **this repo** |
| Wavelet+NN | 0.757(29) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|

 #### 6. PTB-XL: Rhythm statements

| Model | AUC &darr; | paper/source | code |
|---:|:---|:---|:---|
| xresnet1d101 | 0.957(19) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| inception1d | 0.953(13) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| lstm | 0.953(09) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| lstm_bidir | 0.949(11) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| resnet1d_wang | 0.946(10) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| fcn_wang | 0.931(08) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|
| **Inception Time** | **0.923(32)** | **our work** | **this repo** |
| Wavelet+NN | 0.890(24) | [original work](https://doi.org/10.1109/jbhi.2020.3022989) | [code](https://github.com/helme/ecg_ptbxl_benchmarking/)|



# References
Please acknowledge our work by citing our journal paper

    @article{Strodthoff:2020Deep,
    doi = {10.1109/jbhi.2020.3022989},
    url = {https://doi.org/10.1109/jbhi.2020.3022989},
    year = {2021},
    volume={25},
    number={5},
    pages={1519-1528},
    publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
    author = {Nils Strodthoff and Patrick Wagner and Tobias Schaeffter and Wojciech Samek},
    title = {Deep Learning for {ECG} Analysis: Benchmarks and Insights from {PTB}-{XL}},
    journal = {{IEEE} Journal of Biomedical and Health Informatics}
    }

For the PTB-XL dataset, please cite

    @article{Wagner:2020PTBXL,
    doi = {10.1038/s41597-020-0495-6},
    url = {https://doi.org/10.1038/s41597-020-0495-6},
    year = {2020},
    publisher = {Springer Science and Business Media {LLC}},
    volume = {7},
    number = {1},
    pages = {154},
    author = {Patrick Wagner and Nils Strodthoff and Ralf-Dieter Bousseljot and Dieter Kreiseler and Fatima I. Lunze and Wojciech Samek and Tobias Schaeffter},
    title = {{PTB}-{XL},  a large publicly available electrocardiography dataset},
    journal = {Scientific Data}
    }

    @misc{Wagner2020:ptbxlphysionet,
    title={{PTB-XL, a large publicly available electrocardiography dataset}},
    author={Patrick Wagner and Nils Strodthoff and Ralf-Dieter Bousseljot and Wojciech Samek and Tobias Schaeffter},
    doi={10.13026/qgmg-0d46},
    year={2020},
    journal={PhysioNet}
    }

    @article{Goldberger2020:physionet,
    author = {Ary L. Goldberger  and Luis A. N. Amaral  and Leon Glass  and Jeffrey M. Hausdorff  and Plamen Ch. Ivanov  and Roger G. Mark  and Joseph E. Mietus  and George B. Moody  and Chung-Kang Peng  and H. Eugene Stanley },
    title = {{PhysioBank, PhysioToolkit, and PhysioNet}},
    journal = {Circulation},
    volume = {101},
    number = {23},
    pages = {e215-e220},
    year = {2000},
    doi = {10.1161/01.CIR.101.23.e215}
    }

If you use the [ICBEB challenge 2018 dataset](http://2018.icbeb.org/Challenge.html) please acknowledge

    @article{liu2018:icbeb,
    doi = {10.1166/jmihi.2018.2442},
    year = {2018},
    month = sep,
    publisher = {American Scientific Publishers},
    volume = {8},
    number = {7},
    pages = {1368--1373},
    author = {Feifei Liu and Chengyu Liu and Lina Zhao and Xiangyu Zhang and Xiaoling Wu and Xiaoyan Xu and Yulin Liu and Caiyun Ma and Shoushui Wei and Zhiqiang He and Jianqing Li and Eddie Ng Yin Kwee},
    title = {{An Open Access Database for Evaluating the Algorithms of Electrocardiogram Rhythm and Morphology Abnormality Detection}},
    journal = {Journal of Medical Imaging and Health Informatics}
    }
