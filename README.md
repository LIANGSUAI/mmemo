# mmemo - 毫米波雷达情感识别论文复现

复现论文: **"An emotion recognition dataset using millimeter wave radar and physiological reference signals"**
- 作者: Cai, J., Zhang, X., Pan, Y. & Zhou, H.
- 期刊: Scientific Data (2026)
- DOI: https://doi.org/10.1038/s41597-026-07159-6

## 项目结构

```
├── run_pipeline.py           # 主入口 (一键运行)
├── requirements.txt          # Python依赖
├── src/
│   ├── config.py             # 全局配置 (路径/参数)
│   ├── mmwave_processor.py   # mmWave信号处理
│   ├── ppg_gsr_processor.py  # PPG/GSR信号处理
│   ├── feature_utils.py      # 特征提取工具函数
│   └── extract_features.py   # 特征提取主脚本
└── results/                  # 复现结果 (本地)
```

## 复现内容

### 信号处理
| 信号 | 处理方法 |
|------|---------|
| mmWave | Range-FFT → MTI杂波抑制 → 多bin融合(K=5) → 带通滤波分离呼吸/心跳 |
| PPG | 4阶Butterworth带通滤波(0.6-5Hz) → 去趋势 → 移动平均平滑 |
| GSR | 电阻→电导转换 → 3阶Butterworth低通滤波(1Hz) |

### 特征提取
| 信号 | 维度 | 特征 |
|------|------|------|
| mmWave | 32 | 统计 + 物理(NSI/HFD) + PSD + HHT + HRV |
| PPG | 28 | 统计 + 物理 + PSD + HHT + HRV |
| GSR | 24 | 时域统计(原始/一阶/二阶导数) + PSD |

### 分类
- SVM (RBF核), 75/25划分, GridSearchCV调参
- 二分类: SAM评分 >5=High, <5=Low

## 复现结果

| Signal | Valence Acc | Arousal Acc | Dominance Acc |
|--------|------------|-------------|---------------|
| PPG | 64.8% | 67.2% | 65.1% |
| mmWave | 59.5% | 67.1% | 66.4% |
| GSR | 64.9% | 67.8% | 67.8% |

与论文Table 7差异在 ±3% 以内。

## 使用方法

```bash
pip install -r requirements.txt
# 修改 src/config.py 中的 RAW_DATA_DIR 为数据集路径
python -u run_pipeline.py
```

## 环境要求
- Python 3.10+
- numpy, scipy, scikit-learn, pandas, PyEMD (可选)
