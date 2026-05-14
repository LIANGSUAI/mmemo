"""mmWave雷达信号处理 - 移植自 mmwave_process.m

优化版: 向量化FFT + 断点续跑

流水线:
1. 去掉前10秒注视十字
2. 重塑数据为 [nFrame, nADC, virtualRx]
3. 修正Rx2/Rx3的180度相位差
4. 对每个虚拟天线做 Range-FFT -> extractPhase
5. 去趋势 + 多通道平均
6. 一阶差分
7. 带通滤波分离呼吸 (0.1-0.5Hz) 和心跳 (1.0-1.8Hz)
"""

import os
import sys
import numpy as np
from scipy.signal import butter, filtfilt, detrend
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    NUM_TX, NUM_RX, NUM_ADC_SAMPLES, MMWAVE_FS
)


def extract_phase_vectorized(adc_data_all, n_range_fft=256):
    """向量化版: 同时处理所有12个虚拟天线的相位提取

    Args:
        adc_data_all: shape (12, nADC, nChirps)
        n_range_fft: Range-FFT点数

    Returns:
        phase_channels: shape (nChirps, 12)
    """
    n_rx, n_adc, n_chirps = adc_data_all.shape

    # Range-FFT: 对所有天线同时做 (向量化)
    fft_data = np.fft.fft(adc_data_all, n=n_range_fft, axis=1)  # (12, nRange, nChirps)
    fft_data = fft_data.transpose(0, 2, 1)  # (12, nChirps, nRange)

    phase_channels = np.zeros((n_chirps, n_rx))

    for ch in range(n_rx):
        data_ch = fft_data[ch]  # (nChirps, nRange)

        # MTI: 均值相消
        data_mti = data_ch - np.mean(data_ch, axis=0, keepdims=True)

        # 多bin融合: K=5
        K = 5
        amp = np.mean(np.abs(data_mti), axis=0)
        idx = np.argsort(amp)[-K:]
        weights = amp[idx]
        weights = weights / (np.sum(weights) + 1e-20)

        # 相位融合
        phase_all = np.unwrap(np.angle(data_mti[:, idx]), axis=0)
        phase_channels[:, ch] = phase_all @ weights

    return phase_channels


def bandpass_filter(x, fs, f1, f2, order):
    """带通滤波"""
    Wn = [f1 / (fs / 2), f2 / (fs / 2)]
    n = order // 2
    b, a = butter(n, Wn, btype='bandpass')
    y = filtfilt(b, a, x)
    return y


def process_single_mmwave(adc_data):
    """处理单个mmWave文件 (优化版)

    Args:
        adc_data: shape (12, n_cols), complex128

    Returns:
        vital, respiration, heartbeat
    """
    virtual_rx = NUM_TX * NUM_RX  # 12

    # 去掉前10秒
    n_skip = MMWAVE_FS * NUM_ADC_SAMPLES * 10
    adc_data = adc_data[:, n_skip:]

    # 重塑为 (12, nADC, nFrame) - 向量化
    n_cols = adc_data.shape[1]
    n_frame = n_cols // NUM_ADC_SAMPLES
    data = adc_data[:, :n_frame * NUM_ADC_SAMPLES].reshape(
        virtual_rx, n_frame, NUM_ADC_SAMPLES).transpose(0, 2, 1)
    # data shape: (12, nADC, nFrame)

    # 修正Rx2/Rx3的180度相位差
    for j in range(1, virtual_rx, 4):
        data[j] *= -1
        data[j + 1] *= -1

    # 向量化相位提取
    phase_channels = extract_phase_vectorized(data)

    # 去趋势 + 多通道平均
    phase_channels = detrend(phase_channels, axis=0)
    phi_unwraped = np.mean(phase_channels, axis=1)

    # 一阶差分
    phi_diff = np.diff(phi_unwraped)
    phi_diff = np.append(phi_diff, phi_diff[-1])
    phi = phi_diff

    # 带通滤波
    respiration = bandpass_filter(phi, MMWAVE_FS, 0.1, 0.5, 4)
    heartbeat = bandpass_filter(phi, MMWAVE_FS, 1.0, 1.8, 6)

    return phi, respiration, heartbeat


def batch_process_mmwave():
    """批量处理所有mmWave原始数据 (支持断点续跑)"""
    import scipy.io as sio

    raw_dir = os.path.join(RAW_DATA_DIR, 'mmwave')
    out_dir = os.path.join(PROCESSED_DATA_DIR, 'mmwave')
    os.makedirs(out_dir, exist_ok=True)

    from config import PARTICIPANTS, CLIPS

    total = 0
    skipped = 0
    done = 0
    errors = 0

    for pid in PARTICIPANTS:
        pid_dir = os.path.join(raw_dir, pid)
        if not os.path.isdir(pid_dir):
            print(f'[跳过] {pid} 目录不存在', flush=True)
            continue

        for clip in CLIPS:
            total += 1
            mat_path = os.path.join(pid_dir, f'{clip}.mat')
            out_path = os.path.join(out_dir, f'mmwave_{pid}_{clip}.csv')

            # 断点续跳: 已存在的文件跳过
            if os.path.exists(out_path):
                skipped += 1
                continue

            if not os.path.exists(mat_path):
                continue

            try:
                mat = sio.loadmat(mat_path)
                adc_data = mat['adcData']
                vital, resp, heart = process_single_mmwave(adc_data)

                header = 'vital,respiration,heartbeat'
                np.savetxt(out_path,
                           np.column_stack([vital, resp, heart]),
                           delimiter=',', header=header, comments='')
                done += 1
                print(f'  [完成] {pid}_{clip}, 长度={len(vital)}', flush=True)
            except Exception as e:
                errors += 1
                print(f'  [出错] {pid}_{clip}: {e}', flush=True)

    print(f'\nmmWave处理完成: 总计={total}, 新完成={done}, 跳过={skipped}, 出错={errors}', flush=True)


if __name__ == '__main__':
    batch_process_mmwave()
