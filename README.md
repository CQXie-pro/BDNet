# BDNet: Bi-Domain Seasonal-Trend Decomposition for Long-term Time Series Forecasting
This is the official implementation of BDNet.

## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`. 


3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Linear.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.


## Acknowledgement

We extend our heartfelt appreciation to the following GitHub repositories for providing valuable code bases and datasets:

https://github.com/thuml/Time-Series-Library

https://github.com/lss-1138/SparseTSF

https://github.com/VEWOXIC/FITS

https://github.com/plumprc/RTSF

https://github.com/ACAT-SCUT/CycleNet