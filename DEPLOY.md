# 论文复现部署指南

## 论文信息
- **标题**: An emotion recognition dataset using millimeter wave radar and physiological reference signals
- **期刊**: Scientific Data (2026)
- **作者**: Cai, J., Zhang, X., Pan, Y. & Zhou, H.

## 项目结构

```
repro/
├── run_pipeline.py          # 主运行脚本 (入口)
├── requirements.txt         # Python依赖
├── DEPLOY.md               # 本文档
├── src/
│   ├── config.py           # 全局配置 (路径/参数)
│   ├── mmwave_processor.py # mmWave信号处理
│   ├── ppg_gsr_processor.py# PPG/GSR信号处理
│   ├── feature_utils.py    # 特征提取工具函数
│   └── extract_features.py # 特征提取主脚本
├── mmWaveEmoRecDataset/    # 原始仓库 (git clone)
├── processed_data/         # 信号处理输出
├── features/               # 特征输出
└── results/                # 分类结果
```

## 服务器部署步骤

### 1. 上传代码

将 `repro/` 整个目录上传到服务器:
```bash
scp -r repro/ user@server:/path/to/
```

### 2. 准备数据

数据已经在 `emotion/01_raw_data/` 中。需要确保服务器上的路径一致, 或修改 `src/config.py` 中的 `RAW_DATA_DIR`:

```python
# 修改为服务器上的实际路径
RAW_DATA_DIR = '/path/to/emotion/01_raw_data'
```

### 3. 安装依赖

```bash
cd /path/to/repro
conda create -n mmemo python=3.10 -y
conda activate mmemo
pip install -r requirements.txt
```

### 4. 运行复现

**一键运行全部步骤:**
```bash
python run_pipeline.py
```

**分步运行 (推荐, 便于排查问题):**
```bash
# 步骤1: 信号处理 (最耗时, 约30-60分钟)
python run_pipeline.py --step process

# 步骤2: 特征提取 (约15-30分钟)
python run_pipeline.py --step features

# 步骤3: SVM分类 (约10-20分钟)
python run_pipeline.py --step classify

# 步骤4: 结果汇总
python run_pipeline.py --step summary
```

## 复现内容说明

### 信号处理 (对应论文 Data Records > Processed data)

| 信号 | 处理方法 |
|------|---------|
| mmWave | Range-FFT → MTI杂波抑制 → 多bin融合(K=5) → 多通道平均 → 一阶差分 → 带通滤波分离呼吸/心跳 |
| PPG | 4阶Butterworth带通滤波(0.6-5Hz) → 线性去趋势 → 移动平均平滑(窗口20) |
| GSR | 电阻→电导转换 → 3阶Butterworth低通滤波(1Hz) |

### 特征提取 (对应论文 Technical Validation > Emotion recognition > Feature extraction)

| 信号 | 特征维度 | 特征内容 |
|------|---------|---------|
| mmWave | 32 | 统计(6) + 物理(2) + PSD(7) + HHT(8) + HRV时域(8) + HRV非线性(1) |
| PPG | 28 | 统计(6) + 物理(2) + PSD(3) + HHT(8) + HRV时域(8) + HRV非线性(1) |
| GSR | 24 | 原始信号时域统计(6) + 一阶导数统计(6) + 二阶导数统计(6) + PSD统计(6) |

### 分类 (对应论文 Technical Validation > Emotion recognition > Emotion classification)

- **分类器**: SVM (RBF核)
- **标签**: SAM评分, 阈值5进行二分类 (>5=High, <5=Low, =5跳过)
- **划分**: 每个视频的前75%训练, 后25%测试
- **标准化**: Z-score
- **调参**: GridSearchCV (C: 10^-5~10^5, gamma: 10^-4~10)
- **指标**: Accuracy, Balanced Accuracy, F1-Score

### 论文报告的结果 (Table 7)

| Signal | Valence Acc | Arousal Acc | Dominance Acc |
|--------|------------|-------------|---------------|
| PPG | 62.3% | 64.7% | 65.8% |
| mmWave | 62.0% | 69.3% | 67.2% |
| GSR | 64.2% | 67.2% | 66.3% |

## 注意事项

1. **PyEMD库**: HHT特征提取依赖EMD分解。如果PyEMD安装困难, 代码会自动降级为带通滤波近似。
2. **mmWave处理**: 这是最耗时的步骤, 因为需要对每个chirp做FFT。建议在服务器上运行。
3. **路径修改**: 只需修改 `src/config.py` 中的 `RAW_DATA_DIR` 即可适配不同机器。
4. **内存**: 处理mmWave数据时单个文件可能需要几GB内存, 建议至少16GB。
