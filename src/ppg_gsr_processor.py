"""PPG和GSR信号处理 - 移植自 ppg_process.m 和 gsr_process.m

PPG处理流水线:
1. 去掉前10秒注视十字
2. 4阶Butterworth带通滤波 (0.6-5 Hz)
3. 线性去趋势 (detrend)
4. 移动平均平滑 (窗口20样本)

GSR处理流水线:
1. 去掉前10秒注视十字
2. 电阻(kΩ) -> 电导(μS): SC = 1000 / R
3. 3阶Butterworth低通滤波 (截止1 Hz)
"""

import os
import numpy as np
from scipy.signal import butter, filtfilt
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PPG_GSR_FS, PARTICIPANTS, CLIPS


def process_ppg(raw_ppg, fs=200):
    """处理单个PPG信号 (移植自 ppg_process.m)

    Args:
        raw_ppg: 原始PPG数据, 1D array
        fs: 采样率

    Returns:
        处理后的PPG信号
    """
    # 1. 带通滤波 0.6-5 Hz, 4阶
    order = 4
    if len(raw_ppg) >= 3 * (2 * order) + 1:
        Wn = [0.6 / (fs / 2), 5.0 / (fs / 2)]
        b, a = butter(order, Wn, btype='bandpass')
        filtered = filtfilt(b, a, raw_ppg)
    else:
        filtered = raw_ppg

    # 2. 去趋势
    detrended = filtered - np.polyval(np.polyfit(np.arange(len(filtered)), filtered, 1),
                                       np.arange(len(filtered)))

    # 3. 移动平均平滑 (窗口20)
    window_size = 20
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(detrended, kernel, mode='same')

    return smoothed


def process_gsr(raw_gsr, fs=200):
    """处理单个GSR信号 (移植自 gsr_process.m)

    Args:
        raw_gsr: 原始GSR数据 (电阻kΩ), 1D array
        fs: 采样率

    Returns:
        处理后的GSR信号 (电导μS)
    """
    # 电阻 -> 电导: SC(μS) = 1000 / R(kΩ)
    conductance = np.where(raw_gsr > 0, 1000.0 / raw_gsr, np.nan)

    # 用中位数填充NaN
    valid = conductance[~np.isnan(conductance)]
    if len(valid) > 0:
        median_val = np.median(valid)
        conductance = np.where(np.isnan(conductance), median_val, conductance)

    # 3阶Butterworth低通滤波, 截止1 Hz
    order = 3
    if len(conductance) >= 3 * order + 1:
        Wn = 1.0 / (fs / 2)
        b, a = butter(order, Wn, btype='low')
        filtered = filtfilt(b, a, conductance)
    else:
        filtered = conductance

    return filtered


def batch_process_ppg():
    """批量处理所有PPG数据"""
    raw_dir = os.path.join(RAW_DATA_DIR, 'ppg_and_gsr')
    out_dir = os.path.join(PROCESSED_DATA_DIR, 'ppg')
    os.makedirs(out_dir, exist_ok=True)

    for pid in PARTICIPANTS:
        pid_dir = os.path.join(raw_dir, pid)
        if not os.path.isdir(pid_dir):
            continue
        print(f'[PPG] {pid}')
        for clip in CLIPS:
            csv_path = os.path.join(pid_dir, f'{clip}.csv')
            if not os.path.exists(csv_path):
                continue
            try:
                data = np.loadtxt(csv_path, skiprows=2, delimiter=',')
                raw_ppg = data[PPG_GSR_FS * 10:, 0]  # 去掉前10秒
                processed = process_ppg(raw_ppg)

                out_path = os.path.join(out_dir, f'ppg_{pid}_{clip}.csv')
                np.savetxt(out_path, processed, fmt='%.6f')
                print(f'  -> {clip} 完成, 长度={len(processed)}')
            except Exception as e:
                print(f'  -> {clip} 出错: {e}')


def batch_process_gsr():
    """批量处理所有GSR数据"""
    raw_dir = os.path.join(RAW_DATA_DIR, 'ppg_and_gsr')
    out_dir = os.path.join(PROCESSED_DATA_DIR, 'gsr')
    os.makedirs(out_dir, exist_ok=True)

    for pid in PARTICIPANTS:
        pid_dir = os.path.join(raw_dir, pid)
        if not os.path.isdir(pid_dir):
            continue
        print(f'[GSR] {pid}')
        for clip in CLIPS:
            csv_path = os.path.join(pid_dir, f'{clip}.csv')
            if not os.path.exists(csv_path):
                continue
            try:
                data = np.loadtxt(csv_path, skiprows=2, delimiter=',')
                raw_gsr = data[PPG_GSR_FS * 10:, 1]  # 去掉前10秒, 第2列
                processed = process_gsr(raw_gsr)

                out_path = os.path.join(out_dir, f'gsr_{pid}_{clip}.csv')
                np.savetxt(out_path, processed, fmt='%.6f')
                print(f'  -> {clip} 完成, 长度={len(processed)}')
            except Exception as e:
                print(f'  -> {clip} 出错: {e}')


if __name__ == '__main__':
    batch_process_ppg()
    batch_process_gsr()
