# UFATD

## Introduction

Railway detection is crucial for railway anomaly detection, and we propose a new railway database and a row-based railway detection method.

## SRail

## UFATD

We propose an efficient row-based track detection method UFATD, which consists of a convolutional backbone and two classifiers. One for identifying the row coordinates of rail tracks and another for determining the column coordinates through anchor categorization.

### **train scripts**

```bash
git clone https://github.com/idnihai/UFATD.git
pip install -r requirements.txt
python train.py configs/srail.py
```

### **test scripts**

Download the model and modify the **test_model** and **data_root** in the **configs\dlrail.py** file.

```bash
python test.py configs/dlrail.py
```

## Trained models

| Dataset                                      | F1    | Model                                                                                          | Note                                                                   |
| -------------------------------------------- | ----- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [DL-rail](https://www.alipan.com/s/n1HV3tFpWCF) | 94.67 | [resnet18](https://drive.google.com/file/d/1i34RoZ7Q2LzT1AhDthAfE7p0HlGkRCmv/view?usp=drive_link) | "ego_left" and "ego_right" in the label file (txt) need to be removed. |
|                                              |       |                                                                                                |                                                                        |

## Acknowledgement

Many thanks to the authors of [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection), **[Rail-Detection](https://github.com/Sampson-Lee/Rail-Detection)** and [mmLaneDet](https://github.com/Yzichen/mmLaneDet).
