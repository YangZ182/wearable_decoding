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

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 直接打开交互工具

仓库已包含一份与当前人工反馈匹配的 baseline 测试集预测结果，因此可以不重新训练，直接运行：

```bash
streamlit run human_feedback/app.py
```

打开页面后，可以点击混淆矩阵中的任意类别组合，查看对应测试样本的 9 通道时序信号；多个类别组合可以同时保留比较。错误分类样本下方可以保存人工审阅标签和备注。

## 重新训练模型

如需重新生成训练输出、测试集预测和模型 checkpoint：

```bash
python src/train_cnn1d.py
```

训练输出会保存到新的时间戳目录，不会覆盖当前交互工具默认读取的 baseline 结果：

```text
outputs/training_runs/cnn1d_retrain_<timestamp>/
```

包括：

```text
training_log.csv
confusion_matrix.csv
metrics.json
cnn1d_test_predictions.npz
model_state_dict.pt
```

## 人工反馈输出

交互工具保存的人工审阅结果位于：

```text
outputs/human_feedback/error_review_annotations.csv
```

该文件与训练阶段输出分开保存，便于区分模型自动评估结果和人工审阅结果。当前这份人工反馈对应 `outputs/training_runs/cnn1d_baseline/` 中的预测结果。

