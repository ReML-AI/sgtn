# SGTN: Privacy-Preserving Visual Content Tagging using Graph Transformer Networks
This project implements Privacy-Preserving Visual Content Tagging using Graph Transformer Networks. 

### Requirements
Please, install the following packages
- numpy
- pytorch (1.*)
- torchnet
- torchvision
- tqdm

### Download best checkpoints
- SGTN on MS-COCO - checkpoint/coco/SGTN_N_86.6440.pth.tar ([GDrive](https://drive.google.com/file/d/1kksQ0NGp38XHC5_ZoRinj2E8a6tTyb4Q/view?usp=sharing))
- SGTN on PP-MS-COCO - checkpoint/coco/SGTN_A_85.5768.pth.tar ([GDrive](https://drive.google.com/file/d/11O3Dkbtex4cT-u-MKj5sZkO4DaLO5Gwa/view?usp=sharing))

### Performance

<div id="tbl:mlgcn-comparison">

| Method              |   mAP    |    CP    |    CR    |   CF1    |    OP    |    OR    |   OF1    |
| :------------------ | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| CNN-RNN             |   61.2   |    \-    |    \-    |    \-    |    \-    |    \-    |    \-    |
| SRN                 |   77.1   |   81.6   |   65.4   |   71.2   |   82.7   |   69.9   |   75.8   |
| Baseline(ResNet101) |   77.3   |   80.2   |   66.7   |   72.8   |   83.9   |   70.8   |   76.8   |
| Multi-Evidence      |    –     |   80.4   |   70.2   |   74.9   |   85.2   |   72.5   |   78.4   |
| ML-GCN              |   82.4   | **84.4** |   71.4   |   77.4   | **85.8** |   74.5   | **79.8** |
| SGTN                | **86.6** |   77.2   | **82.2** | **79.6** |   76.0   | **82.6** |   77.2   |
| ML-GCN  (PP)          |   80.3   |   84.6   |   68.1   |   75.5   |   85.2   |   72.4   |   78.3   |
| SGTN (PP)             | **85.6** | **85.3** | **75.3** | **79.9** | **85.3** | **78.7** | **81.8** |

Performance comparisons on COCO and PP-COCO. SGTN outperforms baselines with large margins. 
PP denotes the use of anonymised dataset.

</div>

### TGCN on COCO

```sh
python sgtn.py data/coco --image-size 448 --workers 8 --batch-size 32 --lr 0.03 --learning-rate-decay 0.1 --epoch_step 80 --embedding data/coco/coco_glove_word2vec.pkl --adj_dd_threshold 0.4 --device_ids 0
```

## How to cite this work?

```
@inproceedings{Vu:ACMMM:2020,
	author = {Vu, Xuan-Son and Le, Duc-Trong and Edlund, Christoffer and Jiang, Lili and Nguyen, Hoang D.},
	title = {Privacy-Preserving Visual Content Tagging using Graph Transformer Networks},
	booktitle = {ACM International Conference on Multimedia},
	series = {ACM MM '20},
	year = {2020},
	publisher = {ACM},
	address = {New York, NY, USA}
}
```


## Reference
This project is based on the following implementations:

- https://github.com/durandtibo/wildcat.pytorch
- https://github.com/tkipf/pygcn
- https://github.com/Megvii-Nanjing/ML_GCN/
- https://github.com/seongjunyun/Graph_Transformer_Networks


