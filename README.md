# wearable_decoding

这是一个基于 UCI HAR 数据集的人体活动识别模型诊断项目。项目包含 1D CNN 训练脚本、原始数据、测试集预测结果，以及一个 Streamlit 人工反馈工具，用来从混淆矩阵下钻到原始 9 通道惯性时序信号，并记录人工审阅结果。

## 项目结构

```text
data/                         # UCI HAR 原始数据
src/                          # 1D CNN 训练代码
human_feedback/               # Streamlit 交互式诊断工具
outputs/training_runs/         # 训练与测试阶段输出
outputs/human_feedback/        # 人工审阅结果
tests/                        # 简单自检
```

## 数据来源

本项目使用 UCI Human Activity Recognition Using Smartphones Dataset。原始数据位于 `data/`，数据许可和引用信息见 `DATA_LICENSE.md`。

## 安装并启动交互 Demo

在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
python -m streamlit run human_feedback/app.py
```

仓库已包含一份与当前人工反馈匹配的 baseline 测试集预测结果，因此不需要重新训练。当前 Windows + Anaconda `base` 环境已用 Python 3.12.7 验证；这台电脑需要在启动前忽略用户目录里冲突的 Python 包：

```powershell
$env:PYTHONNOUSERSITE="1"
python -m streamlit run human_feedback/app.py
```

打开页面后，可以点击混淆矩阵中的任意类别组合，查看对应测试样本的 9 通道时序信号；多个类别组合可以同时保留比较。错误分类样本下方可以保存人工审阅标签和备注。

## 部署到 Streamlit Community Cloud

先把本地提交推送到 GitHub，然后在 [Streamlit Community Cloud](https://share.streamlit.io) 创建应用，填写：

```text
Repository: YangZ182/wearable_decoding
Branch: main
Main file path: human_feedback/app.py
Python version: 3.12
```

本项目不需要 Secrets、`packages.txt` 或额外 Streamlit 配置。App URL 可在可用时设为 `wearable-decoding`，对应简历链接形式为：

```text
https://wearable-decoding.streamlit.app
```

网页会把每位访客新增的审阅保存在各自的页面会话中，不会让公开访客相互覆盖，也不会修改仓库内附带的 baseline。刷新、断线或关闭页面前，点击“下载当前审阅 CSV”保留结果。

## 重新训练模型

训练按受试者从原训练集固定划分验证集（split seed 42），按验证集 Macro-F1 保存最佳 checkpoint；训练期间不读取测试集。本机训练解释器是 `C:\Users\win11\.conda\envs\wearable\python.exe`，使用 CPU：

```powershell
& C:/Users/win11/.conda/envs/wearable/python.exe -s -m pip install -r requirements-training.txt
& C:/Users/win11/.conda/envs/wearable/python.exe -s -B -X utf8 src/train_cnn1d.py --epochs 30 --depth 3
```

可选参数包括 `--depth 2/3`、`--scheduler`（验证 loss 停滞后学习率减半）、`--weight-decay` 和训练随机种子 `--seed`。本次按验证集选定三层、30 轮、固定学习率、无 weight decay，最佳 checkpoint 在第 29 轮；测试 Accuracy 89.79%、Macro-F1 89.71%。实验记录保存在 `outputs/training_runs/optimization_summary_20260918.json`。单独评估这份 checkpoint：

```powershell
& C:/Users/win11/.conda/envs/wearable/python.exe -s -B -X utf8 src/train_cnn1d.py --evaluate-checkpoint outputs/training_runs/cnn1d_retrain_20260918T082938099944Z/model_state_dict.pt
```

训练输出会保存到新的时间戳目录，不会覆盖当前交互工具默认读取的 baseline 结果：

```text
outputs/training_runs/cnn1d_retrain_<timestamp>/
```

包括：

```text
training_log.csv
run_info.json
validation_predictions.npz
model_state_dict.pt
```

单独评估测试集后，在同一 run 目录增加：

```text
confusion_matrix.csv
metrics.json
cnn1d_test_predictions.npz
```

## 人工反馈输出

仓库附带的 baseline 人工审阅结果位于：

```text
outputs/human_feedback/error_review_annotations.csv
```

该文件与训练阶段输出分开保存，便于区分模型自动评估结果和人工审阅结果。当前这份人工反馈对应 `outputs/training_runs/cnn1d_baseline/` 中的预测结果；网页将它作为只读初始值，新增审阅通过页面下载保存。

