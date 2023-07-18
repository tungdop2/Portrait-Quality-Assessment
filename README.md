# Portrait-Quality-Assessment
This repository contains the code to train and evaluate different models and methods for portrait quality assessment for [PIQ23 Dataset (CVPR 2023)](https://github.com/DXOMARK-Research/PIQ2023).

## Dataset
- For original dataset, please refer to [PIQ23 Dataset (CVPR 2023)](https://github.com/DXOMARK-Research/PIQ2023).
- After downloading the dataset, please put images in `PIQ23/Images` folder.

## Training
This code is config-based. You can modify the config file to train different models and methods.
Example:
```
python train.py --config configs/resnet/resnet18.yaml
```
Specific config will be automatically merged with `configs/base.yaml`.

## Results

Now supports following backbone:

| Model | Freeze    | Val acc    |
| :---:   | :---: | :---: |
| Resnet18 | False   |    |
| Resnet50 | False   |    |

## TODO
- [ ] Add more backbones
- [ ] Add more methods

## Reference
Thanks to authors of [PIQ23 Dataset (CVPR 2023)](https://github.com/DXOMARK-Research/PIQ2023), they have done a great job!
