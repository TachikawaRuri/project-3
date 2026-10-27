# Project 3 — Fundus 眼底图像多标签分类（ResNet18）

## 环境
Python 3.10，依赖见 `requirements.txt`：

```
pip install -r requirements.txt
```

实测环境为 CPU（12 线程，无 GPU），全部实验在 CPU 上完成，约 40 分钟。

## 数据
使用课程提供的 `student_dataset/`，包含：

```
student_dataset/
├── images/                 # 2208 张 300×300 RGB PNG
├── train.csv               # 1545 张，20 列多标签
├── val.csv                 # 221 张，仅用于模型选择
├── test.csv                # 442 张，仅 path
└── sample_submission.csv   # 提交模板
```

代码默认在当前目录的上一级寻找 `student_dataset/`。若数据在别处，设置环境变量：

```bash
set PROJECT3_DATA=D:\path\to\student_dataset     # Windows
export PROJECT3_DATA=/path/to/student_dataset    # Linux/macOS
```

## 复现命令

```bash
python train.py       --model resnet18 --epochs 20 --out ../runs/r18_base
python predict.py     --ckpt ../runs/r18_base/best.pt --out ../test.csv
python eval_report.py --ckpt ../runs/r18_base/best.pt --outdir ../runs/r18_base/report
```

- `train.py`：训练与逐 epoch 验证（macro / micro AUC），保存验证集 macro AUC 最高的权重
- `predict.py`：加载权重，水平翻转 TTA 推理，按 `sample_submission.csv` 行序写概率 CSV
- `eval_report.py`：逐类 AUC / Precision / Recall / F1，并输出 Grad-CAM 图

## 结果
验证集 macro AUC **0.9264**，micro AUC **0.9426**（第 14 epoch，ResNet18 + ImageNet 预训练）。

## 说明
- 验证集**未参与训练**，仅用于选择最优 epoch 与统计指标。
- 预训练权重：torchvision `ResNet18_Weights.IMAGENET1K_V1`（ImageNet-1K）。
- 未使用多模型集成，符合课程"只需选一个模型"的要求。
