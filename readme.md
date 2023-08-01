# TrajCL: Contrastive Trajectory Similarity Learning with Dual-Feature Attention


This is a pytorch implementation of the [TrajCL paper](https://arxiv.org/pdf/2210.05155.pdf):

```
@inproceedings{chang2023contrastive,
  title={Contrastive Trajectory Similarity Learning with Dual-Feature Attention},
  author={Chang, Yanchuan and Qi, Jianzhong and Liang, Yuxuan and Tanin, Egemen},
  booktitle={2023 IEEE 39th International Conference on Data Engineering (ICDE)},
  pages={2933--2945},
  year={2023},
  organization={IEEE}
}
```


## Requirements
- Ubuntu 20.04 LTS with Python 3.7.7
- `pip install -r requirements.txt`
- Datasets can be downloaded from [here](https://drive.google.com/drive/folders/1wvFSdi4T1RvG1ww7TlobQJoTSBdJ7zWq?usp=sharing), and `tar -zxvf TrajCL_dataset.tar.gz -C ./data/`


## Quick Start
To train TrajCL and test it as a standalone trajectory measure (cf. Section V.B in paper):

```bash
python train.py --dataset porto
```

To fine-tune the pre-trained TrajCL to learn to approximate existing heuristic trajectory similarity measures (cf. Section V.F in paper):

(Prerequisites: a pre-trained TrajCL model, in other words, make sure you ran the last command once.)

```bash
python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name hausdorff
```


## FAQ
#### Installation
It may occur failure while installing torch-geometric related packages, including torch-scatter, torch-sparse, torch-cluster and torch-spline-conv, when you use `pip install xxx` to install them directly. That is a commom issue for the older PyTorch versions. A solution can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). Simply speaking, these package need to be installed from wheels. For example, `pip install torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html`.

#### Datasets
To use your own datasets, you may need to create your own pre-processing script like `./utils/preprocessing_porto.py`. Also, the MBR of the space is required to fill into `config.py`. (See `./utils/preprocessing_porto.py` and `config.py` for more details.)



## Contact
Email changyanchuan@gmail.com if you have any queries.
