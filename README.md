
<div align="center">
<h2>MaSS13K: A Matting-level Semantic Segmentation Benchmark</h2>



Chenxi Xie<sup>1,2</sup>
| Minghan Li<sup>1,2</sup> | 
Hui Zeng<sup>2</sup> | 
Jun Luo<sup>2</sup> | 
Lei zhang<sup>1,2</sup> 

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute

🚩 Accepted by CVPR2025

<a href='https://arxiv.org/abs/2503.18364'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

</div>

![MaSS13K Dataset](fig/mas13k.png)
## ⏰ TODO
- [x] Create repo.
- [x] Release arxiv paper.
- [x] Release MaSS13K Dataset.
- [x] Release code.




## 🌟 MaSS13K Datasets
The dataset now is available :[`GoogleDrive`](https://drive.google.com/drive/folders/1NooQARLS4epj6GTsf_6hzphl1APUdW7t?usp=sharing)

Download the dataset and extract in mmsegmentation/data


## 🌟 Our Framework

![MaSSFormer](fig/Framework.png)
Architecture of MaSSFormer. The model architecture is shown on the top left corner, including the image encoder, pixel decoder and transformer decoder. The detailed structure of the high-resolution pixel decoder is shown on the right of this figure.

## MaSS13K-Benchmark


### Qualitative Comparison
![Visual Comparison](fig/VisualComp.png)
Qualitative comparisons of MaSSFormer with other three baseline methods.

### Quantitative Comparison

![Mass13K-Benchmark](fig/QuantitativeComp.png)
Quantitative evaluation on MaSS13K validation and test sets.

## 1️⃣Enviroment

``` bash
conda create -n massformer python=3.10
conda activate massformer
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv==2.0.0rc2  mmengine==0.10.3  
cd mmsegmentation
pip install -e .
```
## 2️⃣Training
1. Download the dataset.  
2. Extract the dataset and put them under data/
3. Modify the data path in "configs/massformer/massformer_r50_8xb2-90k_mass-1024x1024.py" and "configs/\_base_/datasets/mass13k.py"
```bash
tools/dist_train.sh configs/massformer/massformer_r50_8xb2-90k_mass-1024x1024.py 4
```

## 🌏 Citation

```bash
@InProceedings{Xie2025mass13k,
    author    = {Chenxi Xie, Minghan Li, Hui Zeng, Jun Luo and Zhang, Lei},
    title     = {MaSS13K: A Matting-level Semantic Segmentation Benchmark},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025}
}
```


### License
This project is released under the [Apache 2.0 license](LICENSE).

### Acknowledgement

### Contact
If you have any questions, please contact: chenxi.xie@connect.polyu.hk


<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=xiechenxi99.MaSS13K)

</details>
