"""全局配置 - 论文复现参数"""

import os

# ============================================================
# 路径配置 (部署时修改这里)
# ============================================================
# 项目根目录 (自动检测)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 原始数据目录 (你下载的数据集)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, '..', 'emotion', '01_raw_data')

# 处理后数据输出目录
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_data')

# 特征输出目录
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'features')

# 分类结果目录
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# ============================================================
# mmWave 雷达参数 (论文 Table 2)
# ============================================================
NUM_TX = 3               # 发射天线数
NUM_RX = 4               # 接收天线数
NUM_ADC_SAMPLES = 256    # 每个chirp的ADC采样点数
ADC_FS = 5120e3          # ADC采样率 (Hz)
FRAME_PERIODICITY = 10e-3  # 帧周期 (秒)
MMWAVE_FS = int(1 / FRAME_PERIODICITY)  # 100 Hz
SLOPE = 66.590e12        # 频率斜率 (Hz/s)
START_FREQ = 60e9        # 起始频率 (Hz)

# ============================================================
# PPG / GSR 参数
# ============================================================
PPG_GSR_FS = 200  # 采样率 (Hz)

# ============================================================
# 特征提取参数
# ============================================================
SEC_FOR_FEATURE = 60  # 取最后60秒做特征提取

# mmWave: 5秒窗口, 无重叠
MMWAVE_WIN_SIZE = round(MMWAVE_FS * 5)    # 500
MMWAVE_WIN_STRIDE = round(MMWAVE_FS * 5)  # 500

# PPG: 5秒窗口, 无重叠
PPG_WIN_SIZE = round(PPG_GSR_FS * 5)    # 1000
PPG_WIN_STRIDE = round(PPG_GSR_FS * 5)  # 1000

# GSR: 5秒窗口, 无重叠
GSR_WIN_SIZE = round(PPG_GSR_FS * 5)    # 1000
GSR_WIN_STRIDE = round(PPG_GSR_FS * 5)  # 1000

# ============================================================
# 参与者和片段列表
# ============================================================
PARTICIPANTS = [f'P{i:02d}' for i in range(1, 16)]
CLIPS = [f'{i:02d}' for i in range(0, 19)]  # 00=baseline, 01-18=clips
