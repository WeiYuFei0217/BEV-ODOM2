# BEV-ODOM2

**BEV-ODOM2: Enhanced BEV-based Monocular Visual Odometry with PV-BEV Fusion and Dense Flow Supervision for Ground Robots**

[English](README.md) | 中文

[![arXiv](https://img.shields.io/badge/arXiv-BEV--ODOM2-red)](https://arxiv.org/)
[![Project](https://img.shields.io/badge/Project-BEV--ODOM2-blue)](https://github.com/WeiYuFei0217/BEV-ODOM2)
[![ZJH-VO Dataset](https://img.shields.io/badge/Dataset-ZJH--VO-green)](https://github.com/WeiYuFei0217/ZJH-VO-Dataset/)

## 目录

- [简介](#简介)
  - [框架结构](#框架结构)
  - [案例分析](#案例分析)
  - [可视化](#可视化)
- [环境配置](#环境配置)
- [数据集准备](#数据集准备)
- [训练](#训练)
- [测试](#测试)
- [预训练模型](#预训练模型)
- [引用](#引用)
- [致谢](#致谢)

## 简介

<p align="center">
  <img src="./README/figs/fig1.png" width="90%" />
</p>
<p align="center"><i>BEV 单目视觉里程计方法对比。已有 BEV MVO 方法需要额外传感器/标注以实现稠密监督，而 BEV-ODOM 仅使用稀疏位姿标签。BEV-ODOM2（本文方法）通过 PV-BEV 双分支融合和 BEV 光流实现了无需额外标注的稠密监督。</i></p>

鸟瞰图（BEV）表示提供了一个具有度量尺度的平面工作空间，有助于将6自由度自身运动简化为更鲁棒的3自由度模型，适用于智能交通系统中的单目视觉里程计（MVO）。然而，现有的BEV方法存在**稀疏监督信号**和透视图到BEV投影过程中的**信息损失**两个关键问题。

**BEV-ODOM2** 是一个增强的框架，在无需额外标注的情况下同时解决了这两个局限：

1. **稠密BEV光流监督** — 直接从3自由度位姿变换构建像素级光流真值，仅利用已有的位姿数据即可提供稠密的对应关系指导。
2. **PV-BEV双分支融合** — 在LSS投影之前，在透视图（PV）中计算相关性体积以保留6自由度运动线索，然后将其与BEV原生相关性特征融合，实现全面的运动理解。
3. **增强旋转采样** — 采用概率采样策略，平衡高旋转和标准旋转训练样本对，有效应对数据集中直线行驶偏置的问题。

该框架仅从位姿数据中派生出三个层级的监督信号：稠密BEV光流、PV分支的5自由度监督以及最终的3自由度输出。在 **KITTI**、**NCLT**、**Oxford RobotCar** 以及我们新采集的 **ZJH-VO** 多尺度数据集上的大量实验表明，BEV-ODOM2达到了最先进的性能，相较于先前的BEV方法在RTE指标上提升了 **40%**。

### 框架结构

<p align="center">
  <img src="./README/figs/framework.png" width="90%" />
</p>
<p align="center"><i>BEV-ODOM2 框架总览，包括共享特征提取的 PV-BEV 编码器、用于 5-DoF 监督的 PV 相关性分支、通过 FlowUNet 进行稠密光流预测的 BEV 相关性模块，以及最终的 3-DoF 位姿回归。</i></p>

### 案例分析

<p align="center">
  <img src="./README/figs/case_study.png" width="100%" />
</p>
<p align="center"><i>案例分析：展示 NCLT、Oxford、KITTI 和 ZJH-VO 四个数据集上直行（S）、左转（L）和右转（R）典型动作的预测轨迹、稠密 BEV 光流场及误差分析。</i></p>

### 可视化

<p align="center">
  <img src="./README/figs/NCLT&Oxford.gif" width="100%" />
</p>
<p align="center"><i>NCLT 和 Oxford RobotCar 数据集上的轨迹与稠密 BEV 光流可视化。</i></p>

<p align="center">
  <img src="./README/figs/KITTI&ZJH.gif" width="100%" />
</p>
<p align="center"><i>KITTI 和 ZJH-VO 数据集上的轨迹与稠密 BEV 光流可视化。</i></p>

## 环境配置

### 前置要求
- CUDA 11.6
- Python 3.9.18
- PyTorch 1.13.0

### 安装步骤

1. 创建并激活 conda 环境：
```bash
conda create -n bevodom2 python=3.9.18
conda activate bevodom2
```

2. 按以下顺序逐条安装依赖：
```bash
cd ./BEV-ODOM2/
pip install "pip<24.1"
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip uninstall torchmetrics
python setup.py develop
pip install --upgrade networkx
pip install transforms3d
pip install --upgrade pip
pip install spatial-correlation-sampler==0.4.0
```

## 数据集准备

BEV-ODOM2 在以下数据集上进行了评估：

| 数据集 | 相机数量 | 描述 |
|--------|---------|------|
| [NCLT](http://robots.engin.umich.edu/nclt/) | 5（单目：前置相机） | 密歇根大学北校区长期数据集，包含车辆抖动、光照变化和季节差异 |
| [Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/) | 3（单目：后置相机） | 复杂城市驾驶场景，含不同交通密度和动态物体 |
| [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) | 1 | 标准自动驾驶基准数据集，含海拔变化 |
| [ZJH-VO](https://github.com/WeiYuFei0217/ZJH-VO-Dataset/) | 4（单目：左前相机） | 我们新采集的多场景、多尺度数据集，覆盖地下车库、室外广场、走廊和办公区 |

<p align="center">
  <img src="./README/figs/ZJH-VO_dataset_intro.png" width="100%" />
</p>
<p align="center"><i>ZJH-VO 数据集概览：数据采集平台（左）、多楼层轨迹（中）、代表性场景包括办公区、走廊、室外广场和地下车库（右）。</i></p>

### 目录结构

请按照以下结构组织数据集，并在配置文件中更新对应路径（见 [训练](#训练)）：

```
<your_data_root>/
├── NCLT/
│   └── format_data/
│       ├── 2013-04-05/
│       │   ├── lb3_u_s_384/
│       │   └── ground_truth/
│       ├── 2012-01-08/
│       ├── ...
│       └── image_meta.pkl
├── oxford/
│   ├── 2019-01-11-13-24-51/
│   ├── 2019-01-14-14-15-12/
│   ├── ...
│   └── image_meta.pkl
├── kitti/
│   ├── 00/
│   ├── 01/
│   ├── ...
│   └── 10/
└── ZJH/
    └── bag_data/
```

## 训练

1. **配置数据集路径**：打开 `bevodom2/config_files/` 下的配置文件，更新数据集路径：

```yaml
# 在 bevodom2/config_files/NCLT.yaml（或 Oxford.yaml）中
training_params:
  data_root_nclt: "/your/path/to/NCLT/format_data"
  data_root_oxford: "/your/path/to/oxford"
```

2. **（可选）启动 TensorBoard 进行训练监控**：
```bash
tensorboard --logdir=./bevodom2/outputs/ --samples_per_plugin=images=100
```

3. **开始训练**：
```bash
cd ./BEV-ODOM2/bevodom2/train_model/
python train.py --config=../config_files/NCLT.yaml --gpu=0
```

可用配置文件：
- `NCLT.yaml` — NCLT 数据集
- `Oxford.yaml` — Oxford RobotCar 数据集

支持多 GPU 训练：
```bash
python train.py --config=../config_files/NCLT.yaml --gpu=0,1,2
```

## 测试

1. **在配置文件中设置模型路径**：
```yaml
training_params:
  JUST_TEST: true
  ckpt_path: "/path/to/your/checkpoint.pth"
```

2. **运行评估**：
```bash
cd ./BEV-ODOM2/bevodom2/train_model/
python train.py --config=../config_files/NCLT.yaml --gpu=0
```

## 预训练模型

我们提供了 NCLT 和 Oxford 数据集的预训练骨干模型及最终训练模型。

所有模型文件下载地址：[百度网盘](https://pan.baidu.com/s/1bENj0eRTGMiYw5hnZB15Dw?pwd=ODOM)（提取码：`ODOM`）

| 数据集 | Bevdepth预训练权重 | 最终模型 |
|--------|-----------|----------|
| NCLT | ✅ 已包含 | ✅ 已包含 |
| Oxford | ✅ 已包含 | ✅ 已包含 |

## 引用

如果您觉得本工作对您有帮助，请引用：

```bibtex
@article{wei2025bevodom2,
  title={BEV-ODOM2: Enhanced BEV-based Monocular Visual Odometry with PV-BEV Fusion and Dense Flow Supervision for Ground Robots},
  author={Wei, Yufei and Lu, Wangtao and Lu, Sha and Hu, Chenxiao and Han, Fuzhang and Xiong, Rong and Wang, Yue},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025}
}
```

## 致谢

本项目基于以下优秀的开源工作构建：[BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)、[MMDetection3D](https://github.com/open-mmlab/mmdetection3d)、[spatial-correlation-sampler](https://github.com/ClementPinard/Pytorch-Correlation-extension)。同时感谢 [NCLT](http://robots.engin.umich.edu/nclt/)、[Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/) 和 [KITTI](http://www.cvlibs.net/datasets/kitti/) 数据集的创建者。

## 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。
