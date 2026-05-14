"""特征提取主脚本 - 移植自 mmwave_feature.m, ppg_feature.m, gsr_features.m

从 processed_data/ 读取处理后的信号, 提取特征保存到 features/ 目录.
输出格式: .mat 文件, 与原始MATLAB代码兼容.
"""

import os
import re
import numpy as np
import scipy.io as sio
from config import (
    PROCESSED_DATA_DIR, FEATURES_DIR,
    MMWAVE_FS, PPG_GSR_FS,
    SEC_FOR_FEATURE,
    MMWAVE_WIN_SIZE, MMWAVE_WIN_STRIDE,
    PPG_WIN_SIZE, PPG_WIN_STRIDE,
    GSR_WIN_SIZE, GSR_WIN_STRIDE,
    PARTICIPANTS, CLIPS
)
from feature_utils import (
    statistical_feature, physical_feature,
    psd_energy_mmwave, psd_energy_ppg,
    hht_fre_amp, hrv_feature_mmwave, hrv_feature_ppg,
    feature_gsr
)


def extract_mmwave_features():
    """提取mmWave特征 (移植自 mmwave_feature.m)"""
    input_dir = os.path.join(PROCESSED_DATA_DIR, 'mmwave')
    output_dir = os.path.join(FEATURES_DIR, 'mmwave')
    os.makedirs(output_dir, exist_ok=True)

    fs = MMWAVE_FS
    win_size = MMWAVE_WIN_SIZE
    win_stride = MMWAVE_WIN_STRIDE
    sec_for_feature = SEC_FOR_FEATURE

    pattern = re.compile(r'mmwave_(P\d+)_(\d+)\.csv')
    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])

    print(f'[mmWave特征] 找到 {len(csv_files)} 个文件')

    for csv_file in csv_files:
        m = pattern.match(csv_file)
        if not m:
            continue
        pid, clip = m.group(1), m.group(2)

        # 断点续跳
        out_path = os.path.join(output_dir, f'mmwaveFea_{pid}_{clip}.mat')
        if os.path.exists(out_path):
            continue

        try:
            # 读取CSV (有header)
            data = np.genfromtxt(os.path.join(input_dir, csv_file),
                                 delimiter=',', skip_header=1)
            if data.ndim == 1:
                continue

            vital_raw = data[:, 0]
            resp_raw = data[:, 1]
            heart_raw = data[:, 2]

            # 取最后60秒
            samples_to_keep = sec_for_feature * fs
            if len(vital_raw) < samples_to_keep:
                start_idx = 0
            else:
                start_idx = len(vital_raw) - samples_to_keep

            vital = vital_raw[start_idx:]
            resp = resp_raw[start_idx:]
            heart = heart_raw[start_idx:]

            if len(vital) < win_size:
                print(f'  -> {pid}_{clip}: 信号太短, 跳过')
                continue

            # 统计特征
            means, sd, mad, madn, mad2, mad2n = statistical_feature(vital, win_size, win_stride)
            # 物理特征
            nsi, hfd = physical_feature(vital, win_size, win_stride)
            # PSD能量
            resp_psd, heart_psd = psd_energy_mmwave(resp, heart, win_size, win_stride, fs)
            # HHT
            mean_freqs, mean_amp2s = hht_fre_amp(vital, win_size, win_stride, fs, 4)

            # HRV特征 (需要对心跳信号做二阶差分平滑, 与MATLAB一致)
            h = 1.0 / fs
            n = len(heart)
            if n >= 7:
                d2f_core = np.zeros(n - 6)
                for k in range(3, n - 3):
                    num = (4 * heart[k] + (heart[k+1] + heart[k-1])
                           - 2 * (heart[k+2] + heart[k-2]) - (heart[k+3] + heart[k-3]))
                    d2f_core[k - 3] = num / (16 * h ** 2)
                d2f = np.concatenate([
                    np.full(3, d2f_core[0]), d2f_core, np.full(3, d2f_core[-1])
                ])
                # 高斯平滑
                from scipy.ndimage import gaussian_filter1d
                smooth_win = max(1, int(0.2 * fs))
                d2f = gaussian_filter1d(d2f, smooth_win)

                hrv_feats = hrv_feature_mmwave(d2f, win_size, win_stride, fs)
            else:
                n_wins = len(means)
                hrv_feats = [{'meanNN': np.nan, 'medianNN': np.nan, 'SDNN': np.nan,
                              'RMSSD': np.nan, 'PNN50': np.nan, 'meanRate': np.nan,
                              'sdRate': np.nan, 'HRVTi': np.nan, 'PoincareSD1': np.nan}
                             ] * n_wins

            # 组装特征矩阵
            n_wins = len(means)
            # mmWave: 6(统计) + 2(物理) + 4+3(PSD) + 4+4(HHT) + 8(HRV时域) + 1(HRV非线性) = 32
            feature_matrix = np.zeros((n_wins, 32))

            for w in range(n_wins):
                feat = []
                # 统计 (6)
                feat.extend([means[w], sd[w], mad[w], madn[w], mad2[w], mad2n[w]])
                # 物理 (2)
                feat.extend([nsi[w], hfd[w]])
                # PSD (4+3=7)
                feat.extend(resp_psd[:, w].tolist())
                feat.extend(heart_psd[:, w].tolist())
                # HHT (4+4=8)
                feat.extend(mean_freqs[:, w].tolist())
                feat.extend(mean_amp2s[:, w].tolist())
                # HRV时域 (8)
                hf = hrv_feats[w]
                feat.extend([hf['meanNN'], hf['medianNN'], hf['SDNN'], hf['RMSSD'],
                             hf['PNN50'], hf['meanRate'], hf['sdRate'], hf['HRVTi']])
                # HRV非线性 (1)
                feat.append(hf['PoincareSD1'])
                feature_matrix[w, :] = feat

            # 保存
            out_path = os.path.join(output_dir, f'mmwaveFea_{pid}_{clip}.mat')
            sio.savemat(out_path, {'featureMatrix': feature_matrix})
            print(f'  -> {pid}_{clip}: 特征矩阵 shape={feature_matrix.shape}')

        except Exception as e:
            print(f'  -> {pid}_{clip}: 出错 - {e}')


def extract_ppg_features():
    """提取PPG特征 (移植自 ppg_feature.m)"""
    input_dir = os.path.join(PROCESSED_DATA_DIR, 'ppg')
    output_dir = os.path.join(FEATURES_DIR, 'ppg')
    os.makedirs(output_dir, exist_ok=True)

    fs = PPG_GSR_FS
    win_size = PPG_WIN_SIZE
    win_stride = PPG_WIN_STRIDE
    sec_for_feature = SEC_FOR_FEATURE

    pattern = re.compile(r'ppg_(P\d+)_(\d+)\.csv')
    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])

    print(f'[PPG特征] 找到 {len(csv_files)} 个文件')

    for csv_file in csv_files:
        m = pattern.match(csv_file)
        if not m:
            continue
        pid, clip = m.group(1), m.group(2)

        # 断点续跳
        out_path = os.path.join(output_dir, f'ppgFea_{pid}_{clip}.mat')
        if os.path.exists(out_path):
            continue

        try:
            data = np.loadtxt(os.path.join(input_dir, csv_file))
            samples_to_keep = sec_for_feature * fs
            if len(data) < samples_to_keep:
                ppg = data
            else:
                ppg = data[-samples_to_keep:]

            if len(ppg) < win_size:
                continue

            # 统计特征
            means, sd, mad, madn, mad2, mad2n = statistical_feature(ppg, win_size, win_stride)
            # 物理特征
            nsi, hfd = physical_feature(ppg, win_size, win_stride)
            # PSD能量
            heart_psd = psd_energy_ppg(ppg, win_size, win_stride, fs)
            # HHT
            mean_freqs, mean_amp2s = hht_fre_amp(ppg, win_size, win_stride, fs, 4)
            # HRV (基于峰值检测)
            hrv_feats = hrv_feature_ppg(ppg, win_size, win_stride, fs)

            # 组装: 6(统计) + 2(物理) + 3(PSD) + 4+4(HHT) + 8(HRV时域) + 1(HRV非线性) = 28
            n_wins = len(means)
            feature_matrix = np.zeros((n_wins, 28))

            for w in range(n_wins):
                feat = []
                feat.extend([means[w], sd[w], mad[w], madn[w], mad2[w], mad2n[w]])
                feat.extend([nsi[w], hfd[w]])
                feat.extend(heart_psd[:, w].tolist())
                feat.extend(mean_freqs[:, w].tolist())
                feat.extend(mean_amp2s[:, w].tolist())
                hf = hrv_feats[w]
                feat.extend([hf['meanNN'], hf['medianNN'], hf['SDNN'], hf['RMSSD'],
                             hf['PNN50'], hf['meanRate'], hf['sdRate'], hf['HRVTi']])
                feat.append(hf['PoincareSD1'])
                feature_matrix[w, :] = feat

            sio.savemat(out_path, {'featureMatrix': feature_matrix})
            print(f'  -> {pid}_{clip}: 特征矩阵 shape={feature_matrix.shape}', flush=True)

        except Exception as e:
            print(f'  -> {pid}_{clip}: 出错 - {e}', flush=True)


def extract_gsr_features():
    """提取GSR特征 (移植自 gsr_features.m)"""
    input_dir = os.path.join(PROCESSED_DATA_DIR, 'gsr')
    output_dir = os.path.join(FEATURES_DIR, 'gsr')
    os.makedirs(output_dir, exist_ok=True)

    fs = PPG_GSR_FS
    win_size = GSR_WIN_SIZE
    win_stride = GSR_WIN_STRIDE
    sec_for_feature = SEC_FOR_FEATURE

    pattern = re.compile(r'gsr_(P\d+)_(\d+)\.csv')
    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])

    print(f'[GSR特征] 找到 {len(csv_files)} 个文件')

    for csv_file in csv_files:
        m = pattern.match(csv_file)
        if not m:
            continue
        pid, clip = m.group(1), m.group(2)

        try:
            data = np.loadtxt(os.path.join(input_dir, csv_file))
            samples_to_keep = sec_for_feature * fs
            if len(data) < samples_to_keep:
                gsr = data
            else:
                gsr = data[-samples_to_keep:]

            if len(gsr) < win_size:
                continue

            feature_matrix = feature_gsr(gsr, win_size, win_stride, fs)

            out_path = os.path.join(output_dir, f'gsrFea_{pid}_{clip}.mat')
            sio.savemat(out_path, {'featureMatrix': feature_matrix})
            print(f'  -> {pid}_{clip}: 特征矩阵 shape={feature_matrix.shape}')

        except Exception as e:
            print(f'  -> {pid}_{clip}: 出错 - {e}')


if __name__ == '__main__':
    extract_mmwave_features()
    extract_ppg_features()
    extract_gsr_features()
    print('\n所有特征提取完成!')
