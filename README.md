# MRP-GLC
Code for paper: Learning Multiscale Residual Prototypes and Global-Local Correspondence for Video Anomaly Detection

## Datasets
* USCD Ped2 [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]
* CUHK Avenue [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]
* ShanghaiTech [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]

These datasets are from an official github of "Future Frame Prediction for Anomaly Detection - A New Baseline (CVPR 2018)".
Download the datasets into your_dataset_directory.

## Training
```shell
python train.py # for training
```
You can freely define parameters with your own settings like

## Evaluation
Test your own model
Check your dataset_type (ped2, Avenue or shanghai)
```shell
python evaluate.py   # for Evaluation
```

We also provide the pre-trained models and the labels of UCSD Ped2, Avenue and ShanghaiTech datasets at https://pan.baidu.com/s/10_kpqlFrCWmjTJ9NyCSI2Q?pwd=ghak. To test these models, you need download and put them in your folder.

