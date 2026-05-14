"""特征提取工具函数 - 移植自 MATLAB +utils/ 包

包含:
- statisticalFeature: 统计特征 (均值, 标准差, 一阶/二阶差分)
- physicalFeature: 非平稳指数(NSI) + Higuchi分形维数(HFD)
- PSDEnergy_mmwave: mmWave呼吸/心跳PSD能量
- PSDEnergy_ppg: PPG心跳PSD能量
- HHTFreAmp: Hilbert-Huang变换瞬时频率和振幅
- HRVfeature_mmwave: mmWave HRV特征 (基于动态规划的心跳分割)
- HRVfeature_ppg: PPG HRV特征 (基于峰值检测)
- featureGSR: GSR特征 (时域统计 + PSD)
"""

import numpy as np
from scipy.signal import welch, hilbert
from scipy.interpolate import CubicSpline


# ============================================================
# 统计特征 (statisticalFeature.m)
# ============================================================
def statistical_feature(data, win_size, win_stride):
    """计算统计特征: mean, std, 一阶差分绝对值均值, 归一化, 二阶差分, 归一化

    Returns:
        means, sd, mean_abs_diff, mean_abs_diff_nor, mean_abs_diff2, mean_abs_diff2_nor
        每个 shape (num_windows,)
    """
    N = len(data)
    num_windows = (N - win_size) // win_stride + 1

    means = np.zeros(num_windows)
    sd = np.zeros(num_windows)
    mean_abs_diff = np.zeros(num_windows)
    mean_abs_diff_nor = np.zeros(num_windows)
    mean_abs_diff2 = np.zeros(num_windows)
    mean_abs_diff2_nor = np.zeros(num_windows)

    for i in range(num_windows):
        start = i * win_stride
        end = start + win_size
        w = data[start:end]

        means[i] = np.mean(w)
        sd[i] = np.std(w, ddof=1)

        diff_data = np.abs(np.diff(w))
        mean_abs_diff[i] = np.mean(diff_data)
        mean_abs_diff_nor[i] = mean_abs_diff[i] / (sd[i] + 1e-10)

        if len(diff_data) > 1:
            diff_data2 = np.abs(np.diff(diff_data))
            mean_abs_diff2[i] = np.mean(diff_data2)
        else:
            mean_abs_diff2[i] = np.nan
        mean_abs_diff2_nor[i] = mean_abs_diff2[i] / (sd[i] + 1e-10)

    return means, sd, mean_abs_diff, mean_abs_diff_nor, mean_abs_diff2, mean_abs_diff2_nor


# ============================================================
# Higuchi分形维数 (physicalFeature.m 内的 HFD_LCALC)
# ============================================================
def higuchi_fractal_dimension(data, kmax=None):
    """计算Higuchi分形维数"""
    data = np.asarray(data, dtype=float).ravel()
    L = len(data)
    if kmax is None:
        kmax = L // 2
    kmax = max(1, min(kmax, L // 2))

    lastval = np.log10(max(1, L // 2))
    kmax_arr = np.unique(np.round(np.logspace(0.3, lastval, 40)).astype(int))
    kmax_arr = kmax_arr[kmax_arr <= kmax]
    kmax_arr = np.concatenate(([1], kmax_arr))

    LARE = np.zeros(len(kmax_arr))
    for idx, k in enumerate(kmax_arr):
        LAk = 0
        for i in range(k):
            LAi = 0
            n_seg = int(np.floor((L - i - 1) / k))
            for j in range(1, n_seg + 1):
                LAi += abs(data[i + j * k] - data[i + (j - 1) * k])
            if n_seg > 0:
                a = (L - 1) / (n_seg * k)
                LAk += LAi * a / k
        LARE[idx] = LAk / k

    y = np.log(LARE + 1e-20)
    x = np.log(1.0 / kmax_arr)

    # polyfit (跳过第一个点, 与MATLAB一致)
    if len(x) > 2:
        coef = np.polyfit(x[1:], y[1:], 1)
        Df = coef[0]
    else:
        Df = np.nan
    return Df


def physical_feature(data, win_size, win_stride):
    """计算物理特征: 非平稳指数(NSI) + Higuchi分形维数(HFD)"""
    N = len(data)
    num_windows = (N - win_size) // win_stride + 1

    nsi = np.zeros(num_windows)
    hfd = np.zeros(num_windows)

    for i in range(num_windows):
        start = i * win_stride
        end = start + win_size
        w = data[start:end]

        # zscore标准化
        w_norm = (w - np.mean(w)) / (np.std(w, ddof=1) + 1e-10)

        # NSI: 分20段, 计算均值的标准差
        seg_len = int(np.ceil(len(w_norm) / 20))
        seg_means = []
        for j in range(20):
            s = j * seg_len
            e = min((j + 1) * seg_len, len(w_norm))
            if s < e:
                seg_means.append(np.mean(w_norm[s:e]))
        nsi[i] = np.std(seg_means, ddof=1) if len(seg_means) > 1 else 0

        # HFD
        hfd[i] = higuchi_fractal_dimension(w_norm)

    return nsi, hfd


# ============================================================
# PSD能量 (PSDEnergy_mmwave.m, PSDEnergy_ppg.m)
# ============================================================
def psd_energy_mmwave(respiration, heartbeat, win_size, win_stride, fs):
    """计算mmWave呼吸和心跳的PSD频段能量"""
    N = len(heartbeat)
    num_windows = (N - win_size) // win_stride + 1

    fre_seg_res = [0, 0.1, 0.2, 0.3, 0.4]
    fre_seg_hea = [1.0, 1.3, 1.6, 1.8]

    resp_psd = np.zeros((len(fre_seg_res) - 1, num_windows))
    heart_psd = np.zeros((len(fre_seg_hea) - 1, num_windows))

    for i in range(num_windows):
        start = i * win_stride
        end = start + win_size
        w_resp = respiration[start:end]
        w_heart = heartbeat[start:end]

        # 呼吸PSD
        f_r, pxx_r = welch(w_resp, fs=fs, nperseg=min(256, len(w_resp)))
        for j in range(len(fre_seg_res) - 1):
            mask = (f_r >= fre_seg_res[j]) & (f_r < fre_seg_res[j + 1])
            if np.any(mask):
                bw = fre_seg_res[j + 1] - fre_seg_res[j]
                resp_psd[j, i] = np.trapz(pxx_r[mask], f_r[mask]) / bw

        # 心跳PSD
        f_h, pxx_h = welch(w_heart, fs=fs, nperseg=min(256, len(w_heart)))
        for j in range(len(fre_seg_hea) - 1):
            mask = (f_h >= fre_seg_hea[j]) & (f_h < fre_seg_hea[j + 1])
            if np.any(mask):
                bw = fre_seg_hea[j + 1] - fre_seg_hea[j]
                heart_psd[j, i] = np.trapz(pxx_h[mask], f_h[mask]) / bw

    return resp_psd, heart_psd


def psd_energy_ppg(ppg_signal, win_size, win_stride, fs):
    """计算PPG心跳PSD频段能量"""
    N = len(ppg_signal)
    num_windows = (N - win_size) // win_stride + 1

    fre_seg_hea = [1.0, 1.3, 1.6, 1.8]
    heart_psd = np.zeros((len(fre_seg_hea) - 1, num_windows))

    for i in range(num_windows):
        start = i * win_stride
        end = start + win_size
        w = ppg_signal[start:end]

        f_h, pxx_h = welch(w, fs=fs, nperseg=min(256, len(w)))
        for j in range(len(fre_seg_hea) - 1):
            mask = (f_h >= fre_seg_hea[j]) & (f_h < fre_seg_hea[j + 1])
            if np.any(mask):
                bw = fre_seg_hea[j + 1] - fre_seg_hea[j]
                heart_psd[j, i] = np.trapz(pxx_h[mask], f_h[mask]) / bw

    return heart_psd


# ============================================================
# HHT (HHTFreAmp.m)
# ============================================================
def _emd_decompose(data, max_imfs=4):
    """EMD分解 (带超时保护, 超时则用带通滤波替代)"""
    import signal as _signal
    from scipy.signal import butter, filtfilt as _filtfilt

    # 先尝试PyEMD, 设置5秒超时
    try:
        import threading
        result = [None]
        def _run_emd():
            try:
                from PyEMD import EMD
                emd = EMD()
                emd.FIXE = 100
                result[0] = emd.emd(data, max_imfs=max_imfs)[:max_imfs]
            except:
                result[0] = None

        t = threading.Thread(target=_run_emd, daemon=True)
        t.start()
        t.join(timeout=5)
        if t.is_alive():
            print('[EMD] PyEMD超时, 使用带通滤波替代', flush=True)
        elif result[0] is not None and len(result[0]) >= max_imfs:
            return result[0]
    except:
        pass

    # 回退: 带通滤波模拟IMF
    imfs = []
    freq_bands = [(0.1, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, 15.0)]
    for f1, f2 in freq_bands[:max_imfs]:
        try:
            Wn = [f1 / (100 / 2), min(f2 / (100 / 2), 0.99)]
            b, a = butter(3, Wn, btype='bandpass')
            imf = _filtfilt(b, a, data)
            imfs.append(imf)
        except:
            imfs.append(np.zeros_like(data))
    return np.array(imfs)


def _hilbert_stats(imf, fs):
    """计算IMF的Hilbert统计: 平均瞬时频率和平均平方振幅"""
    dt = 1.0 / fs
    analytic = hilbert(imf)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))

    inst_freq = np.diff(phase) / (2 * np.pi * dt)
    inst_freq = np.append(inst_freq, 0)
    inst_freq[inst_freq < 0] = 0

    mean_freq = np.mean(inst_freq)
    mean_amp2 = np.mean(amplitude ** 2)
    return mean_freq, mean_amp2


def hht_fre_amp(data, win_size, win_stride, fs, first_n=4):
    """计算HHT瞬时频率和振幅特征"""
    N = len(data)
    num_windows = (N - win_size) // win_stride + 1

    mean_freqs = np.zeros((first_n, num_windows))
    mean_amp2s = np.zeros((first_n, num_windows))

    for i in range(num_windows):
        start = i * win_stride
        end = start + win_size
        w = data[start:end]

        imfs = _emd_decompose(w, max_imfs=first_n)
        n_actual = min(first_n, len(imfs))

        for j in range(n_actual):
            mf, ma = _hilbert_stats(imfs[j], fs)
            mean_freqs[j, i] = mf
            mean_amp2s[j, i] = ma

    return mean_freqs, mean_amp2s


# ============================================================
# HRV特征 (HRVfeature_mmwave.m, HRVfeature_ppg.m)
# ============================================================
def hrv_feature_from_intervals(interval_ms):
    """从NN间隔(ms)计算HRV特征"""
    if len(interval_ms) < 2:
        nan_dict = {k: np.nan for k in [
            'meanNN', 'medianNN', 'SDNN', 'RMSSD', 'PNN50',
            'meanRate', 'sdRate', 'HRVTi', 'PoincareSD1'
        ]}
        return nan_dict

    mean_nn = np.mean(interval_ms)
    median_nn = np.median(interval_ms)
    sdnn = np.std(interval_ms, ddof=1)

    diffs = np.diff(interval_ms)
    rmssd = np.sqrt(np.mean(diffs ** 2))
    pnn50 = np.sum(np.abs(diffs) > 50) / len(diffs) if len(diffs) > 0 else np.nan

    rate = 60000.0 / interval_ms
    mean_rate = np.mean(rate)
    sd_rate = np.std(rate, ddof=1)

    # HRVTi
    bin_width = 7.8125
    counts, _ = np.histogram(interval_ms, bins=np.arange(
        np.min(interval_ms), np.max(interval_ms) + bin_width, bin_width))
    max_count = np.max(counts) if len(counts) > 0 else 1
    hrv_ti = len(interval_ms) / max(max_count, 1)

    # Poincare SD1
    intervals_sec = interval_ms / 1000.0
    diff_rr = np.diff(intervals_sec)
    poincare_sd1 = np.std(diff_rr, ddof=1) / np.sqrt(2)

    return {
        'meanNN': mean_nn, 'medianNN': median_nn, 'SDNN': sdnn,
        'RMSSD': rmssd, 'PNN50': pnn50, 'meanRate': mean_rate,
        'sdRate': sd_rate, 'HRVTi': hrv_ti, 'PoincareSD1': poincare_sd1
    }


def _interp_template(miu, M, L):
    """将长度为L的模板重采样到M个点 (三次样条, 与MATLAB interp1(...,'spline')一致)"""
    orig_x = np.linspace(0, 1, L)
    query_x = np.linspace(0, 1, M)
    cs = CubicSpline(orig_x, miu, extrapolate=True)
    return cs(query_x)


def _optimal_segmentation(x, miu, b1, b2):
    """动态规划最优心跳分割 (优化版: 线性插值 + 累积和加速)"""
    n = len(x)
    L = len(miu)
    b1, b2 = int(b1), int(b2)

    # 预计算插值模板 (线性插值)
    precomputed = {}
    for M in range(b1, b2 + 1):
        if M > 0:
            precomputed[M] = _interp_template(miu, M, L)

    # 累积和加速cost计算
    x_sq = x ** 2
    cum_x_sq = np.concatenate(([0], np.cumsum(x_sq)))
    cum_x = np.concatenate(([0], np.cumsum(x)))

    D = np.full(n + 1, np.inf)
    D[0] = 0
    prev = np.zeros(n + 1, dtype=int)

    for t in range(1, n + 1):
        min_tau = max(0, t - b2)
        max_tau = t - b1
        if max_tau < min_tau:
            continue

        for tau in range(min_tau, max_tau + 1):
            if tau == t:
                continue
            M = t - tau
            if M not in precomputed:
                continue
            w_miu = precomputed[M]
            # cost = sum((x[tau:t] - w_miu)^2) 用累积和加速
            seg_sq_sum = cum_x_sq[t] - cum_x_sq[tau]
            seg_sum = cum_x[t] - cum_x[tau]
            cost = seg_sq_sum - 2 * np.dot(x[tau:t], w_miu) + np.dot(w_miu, w_miu)
            if D[tau] + cost < D[t]:
                D[t] = D[tau] + cost
                prev[t] = tau

    # 回溯
    segments = []
    pos = n
    while pos > 0:
        p = prev[pos]
        segments.append(x[p:pos])
        pos = p
        if p == 0:
            break
    segments.reverse()
    return segments


def _update_template(segments, m):
    """更新心跳模板 (线性插值)"""
    if not segments:
        return np.zeros(m)

    total_len = sum(len(s) for s in segments)
    new_miu = np.zeros(m)
    for seg in segments:
        L = len(seg)
        if L == 0:
            continue
        new_miu += L * _interp_template(seg, m, L)
    return new_miu / max(total_len, 1)


def _hrv_peak_detection(signal, fs):
    """快速峰值检测回退方案"""
    from scipy.signal import find_peaks as _fp
    min_height = 0.2 * np.max(signal) if np.max(np.abs(signal)) > 0 else 0
    min_dist = int(fs * 0.5)
    peaks, _ = _fp(signal, height=min_height, distance=min_dist)
    if len(peaks) >= 2:
        interval_ms = np.diff(peaks) / fs * 1000
        return hrv_feature_from_intervals(interval_ms)
    return hrv_feature_from_intervals(np.array([]))


def hrv_feature_mmwave(heartbeat, win_size, win_stride, fs):
    """从mmWave心跳信号提取HRV特征 (带超时和峰值检测回退)"""
    import time
    N = len(heartbeat)
    num_windows = (N - win_size) // win_stride + 1

    m = 50
    b1 = fs / 1.8
    b2 = fs / 1.0
    max_iter = 30
    tolerance = 1e-5
    miu_template = np.zeros(m)

    TIMEOUT_PER_WINDOW = 15  # 秒, 超过则切换到峰值检测
    use_dp = True

    all_features = []

    for i in range(num_windows):
        start = i * win_stride
        end = start + win_size
        d2f = heartbeat[start:end]

        if use_dp:
            t0 = time.time()
            miu = miu_template.copy()
            segments = None
            for iteration in range(max_iter):
                miu_old = miu.copy()
                segments = _optimal_segmentation(d2f, miu_old, b1, b2)
                if not segments or any(len(s) == 0 for s in segments):
                    break
                miu = _update_template(segments, m)
                change = np.linalg.norm(miu - miu_old) / (np.linalg.norm(miu_old) + 1e-10)
                if change < tolerance:
                    break
            miu_template = miu.copy()

            elapsed = time.time() - t0
            if elapsed > TIMEOUT_PER_WINDOW:
                print(f'    [警告] HRV窗口{i}耗时{elapsed:.1f}s, 切换为峰值检测', flush=True)
                use_dp = False

            if segments and not any(len(s) == 0 for s in segments):
                interval_samples = np.array([len(s) for s in segments])
                interval_ms = interval_samples / fs * 1000
                feat = hrv_feature_from_intervals(interval_ms)
            else:
                feat = _hrv_peak_detection(d2f, fs)
        else:
            feat = _hrv_peak_detection(d2f, fs)

        all_features.append(feat)

    return all_features


def hrv_feature_ppg(ppg_signal, win_size, win_stride, fs):
    """从PPG信号提取HRV特征 (基于峰值检测)"""
    N = len(ppg_signal)
    num_windows = (N - win_size) // win_stride + 1

    all_features = []

    for i in range(num_windows):
        start = i * win_stride
        end = start + win_size
        w = ppg_signal[start:end]

        # 峰值检测
        try:
            from scipy.signal import find_peaks as _find_peaks
            min_height = 0.2 * np.max(w) if np.max(w) > 0 else 0
            min_dist = int(fs * 0.5)
            peaks, _ = _find_peaks(w, height=min_height, distance=min_dist)
        except:
            peaks = np.array([])

        if len(peaks) >= 2:
            interval_ms = np.diff(peaks) / fs * 1000
            feat = hrv_feature_from_intervals(interval_ms)
        else:
            feat = hrv_feature_from_intervals(np.array([]))

        all_features.append(feat)

    return all_features


# ============================================================
# GSR特征 (featureGSR.m)
# ============================================================
def _time_domain_stats(signal_in):
    """计算时域统计: median, mean, std, min, max, normalized_range"""
    valid = signal_in[~np.isnan(signal_in)]
    if len(valid) == 0:
        return [np.nan] * 6
    med = np.median(valid)
    mean = np.mean(valid)
    std = np.std(valid, ddof=1)
    mn = np.min(valid)
    mx = np.max(valid)
    norm_range = mn / (mx + 1e-10)
    return [med, mean, std, mn, mx, norm_range]


def feature_gsr(data, win_size, win_stride, fs):
    """提取GSR特征: 3×6时域统计 + 6个PSD统计 = 24维"""
    N = len(data)
    num_windows = (N - win_size) // win_stride + 1

    feature_matrix = np.zeros((num_windows, 24))

    for i in range(num_windows):
        start = i * win_stride
        end = start + win_size
        sc = data[start:end]
        sc_diff1 = np.diff(sc)
        sc_diff2 = np.diff(sc, n=2)

        features = []
        features.extend(_time_domain_stats(sc))
        features.extend(_time_domain_stats(sc_diff1))
        features.extend(_time_domain_stats(sc_diff2))

        # PSD特征 (0-2 Hz)
        win_samples = min(int(5 * fs), len(sc))
        if win_samples > 0 and not np.all(np.isnan(sc)):
            try:
                f, pxx = welch(sc, fs=fs, nperseg=min(win_samples, len(sc)),
                               nfft=max(256, int(2 ** np.ceil(np.log2(win_samples)))))
                mask = (f >= 0) & (f <= 2)
                pxx_sel = pxx[mask]
                if len(pxx_sel) > 0:
                    features.extend([
                        np.median(pxx_sel), np.mean(pxx_sel), np.std(pxx_sel),
                        np.max(pxx_sel), np.min(pxx_sel),
                        np.max(pxx_sel) - np.min(pxx_sel)
                    ])
                else:
                    features.extend([np.nan] * 6)
            except:
                features.extend([np.nan] * 6)
        else:
            features.extend([np.nan] * 6)

        feature_matrix[i, :] = features

    return feature_matrix
