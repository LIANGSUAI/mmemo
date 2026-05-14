"""主运行脚本 - 论文复现完整流水线

用法:
    python run_pipeline.py                    # 运行全部步骤
    python run_pipeline.py --step process     # 只运行信号处理
    python run_pipeline.py --step features    # 只运行特征提取
    python run_pipeline.py --step classify    # 只运行分类
    python run_pipeline.py --step summary     # 只运行结果汇总
"""

import argparse
import sys
import os
import time

# 确保 src/ 在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def step_process():
    """步骤1: 信号处理 (mmWave + PPG + GSR)"""
    print('=' * 60)
    print('步骤1: 信号处理')
    print('=' * 60)

    t0 = time.time()

    # mmWave
    print('\n--- mmWave 信号处理 ---')
    from mmwave_processor import batch_process_mmwave
    batch_process_mmwave()

    # PPG & GSR
    print('\n--- PPG 信号处理 ---')
    from ppg_gsr_processor import batch_process_ppg, batch_process_gsr
    batch_process_ppg()

    print('\n--- GSR 信号处理 ---')
    batch_process_gsr()

    print(f'\n信号处理完成, 耗时 {time.time() - t0:.1f} 秒')


def step_features():
    """步骤2: 特征提取"""
    print('=' * 60)
    print('步骤2: 特征提取')
    print('=' * 60)

    t0 = time.time()

    from extract_features import extract_mmwave_features, extract_ppg_features, extract_gsr_features

    print('\n--- mmWave 特征提取 ---')
    extract_mmwave_features()

    print('\n--- PPG 特征提取 ---')
    extract_ppg_features()

    print('\n--- GSR 特征提取 ---')
    extract_gsr_features()

    print(f'\n特征提取完成, 耗时 {time.time() - t0:.1f} 秒')


def step_classify():
    """步骤3: SVM分类 (使用仓库原始classification.py)"""
    print('=' * 60)
    print('步骤3: SVM情感分类')
    print('=' * 60)

    t0 = time.time()

    for feature_type in ['mmwave', 'ppg', 'gsr']:
        print(f'\n--- {feature_type.upper()} 分类 ---')
        run_classification_for_type(feature_type)

    print(f'\n分类完成, 耗时 {time.time() - t0:.1f} 秒')


def run_classification_for_type(feature_type):
    """对指定特征类型运行SVM分类"""
    import re
    import glob
    import numpy as np
    import pandas as pd
    import scipy.io as sio
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    from config import FEATURES_DIR, RESULTS_DIR, RAW_DATA_DIR

    features_data_path = FEATURES_DIR
    sam_data_path = os.path.join(RAW_DATA_DIR, 'self_assessment', 'SAM')
    results_root = os.path.join(RESULTS_DIR, feature_type)
    os.makedirs(results_root, exist_ok=True)

    # 发现参与者
    fea_folder = os.path.join(features_data_path, feature_type)
    if not os.path.exists(fea_folder):
        print(f'  特征目录不存在: {fea_folder}')
        return

    pattern = re.compile(f'{feature_type}Fea_([PS][0-9]+)_')
    all_mat = glob.glob(os.path.join(fea_folder, f'{feature_type}Fea_*.mat'))
    participants = sorted(set(
        pattern.search(os.path.basename(f)).group(1)
        for f in all_mat if pattern.search(os.path.basename(f))
    ))
    print(f'  发现参与者: {participants}')

    for pid in participants:
        for scale in ['valence', 'arousal', 'dominance']:
            print(f'\n  === {pid} - {scale} ===')

            # 加载数据
            sam_path = os.path.join(sam_data_path, f'SAM_{pid}.csv')
            if not os.path.exists(sam_path):
                continue

            sam = pd.read_csv(sam_path, header=0)
            col_idx = {'valence': 1, 'arousal': 2, 'dominance': 3}[scale]
            scores = sam.iloc[:, col_idx].values
            video_ids = sam.iloc[:, 0].values
            labels = np.where(scores > 5, 1, np.where(scores < 5, 0, -1))

            features_list = []
            labels_list = []

            for i, vid in enumerate(video_ids):
                if labels[i] == -1:
                    continue
                vid_str = f'{int(vid):02d}'
                fea_files = glob.glob(os.path.join(
                    fea_folder, f'{feature_type}Fea_{pid}_{vid_str}*.mat'))
                if not fea_files:
                    continue
                try:
                    mat = sio.loadmat(fea_files[0])
                    fm = next((mat[k] for k in mat
                               if not k.startswith('__')
                               and isinstance(mat[k], np.ndarray)
                               and mat[k].ndim == 2), None)
                    if fm is None:
                        continue
                    fm = fm.astype(np.float32)
                    fm = np.nan_to_num(fm, nan=0.0,
                                        posinf=np.finfo(np.float32).max,
                                        neginf=np.finfo(np.float32).min)
                    fm = np.clip(fm, np.finfo(np.float32).min, np.finfo(np.float32).max)
                    features_list.append(fm)
                    labels_list.extend([labels[i]] * fm.shape[0])
                except Exception as e:
                    print(f'    加载出错: {e}')

            if not features_list:
                print(f'    无有效数据')
                continue

            # 75% 训练 / 25% 测试 (按每个视频内部划分)
            X_train_parts, y_train_parts = [], []
            X_test_parts, y_test_parts = [], []
            idx = 0
            for fm in features_list:
                n = fm.shape[0]
                vl = labels_list[idx:idx + n]
                split = int(n * 0.75)
                if 0 < split < n:
                    X_train_parts.append(fm[:split])
                    y_train_parts.extend(vl[:split])
                    X_test_parts.append(fm[split:])
                    y_test_parts.extend(vl[split:])
                elif split == n:
                    X_train_parts.append(fm)
                    y_train_parts.extend(vl)
                else:
                    X_test_parts.append(fm)
                    y_test_parts.extend(vl)
                idx += n

            if not X_train_parts or not X_test_parts:
                print(f'    训练/测试集划分失败')
                continue

            X_train = np.vstack(X_train_parts)
            y_train = np.array(y_train_parts)
            X_test = np.vstack(X_test_parts)
            y_test = np.array(y_test_parts)

            if len(np.unique(y_train)) < 2:
                print(f'    训练集类别不足')
                continue

            # Impute + Scale + SVM
            imputer = SimpleImputer(strategy='mean')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(kernel='rbf', random_state=42,
                            class_weight='balanced', probability=True))
            ])
            param_grid = {
                'svc__C': np.logspace(-5, 5, 11),
                'svc__gamma': np.logspace(-4, 1, 6)
            }
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            gs = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1,
                              scoring='f1_macro', verbose=0)
            try:
                gs.fit(X_train, y_train)
            except ValueError as e:
                print(f'    GridSearch失败: {e}')
                continue

            y_pred = gs.best_estimator_.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            bacc = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            print(f'    Accuracy: {acc:.4f}, Balanced Acc: {bacc:.4f}, F1: {f1:.4f}')

            # 保存结果
            task_dir = os.path.join(results_root, scale, f'{pid}_{feature_type}_{scale}')
            os.makedirs(task_dir, exist_ok=True)
            with open(os.path.join(task_dir, 'ACCnF1.txt'), 'w') as f:
                f.write(f'{acc:.4f}\n{f1:.4f}\n{bacc:.4f}\n')


def step_summary():
    """步骤4: 汇总结果"""
    print('=' * 60)
    print('步骤4: 结果汇总')
    print('=' * 60)

    import glob
    import numpy as np
    from config import RESULTS_DIR

    feature_types = ['ppg', 'mmwave', 'gsr']
    dimensions = ['valence', 'arousal', 'dominance']

    results = {}
    for ft in feature_types:
        results[ft] = {}
        for dim in dimensions:
            pattern = os.path.join(RESULTS_DIR, ft, dim, '*', 'ACCnF1.txt')
            files = glob.glob(pattern)
            accs, f1s, baccs = [], [], []
            for fp in files:
                with open(fp) as f:
                    lines = f.readlines()
                    if len(lines) >= 3:
                        accs.append(float(lines[0].strip()))
                        f1s.append(float(lines[1].strip()))
                        baccs.append(float(lines[2].strip()))
            results[ft][dim] = {
                'acc': np.mean(accs) if accs else 0,
                'f1': np.mean(f1s) if f1s else 0,
                'bacc': np.mean(baccs) if baccs else 0,
            }

    # 打印结果表格
    print('\n' + '=' * 70)
    print('表7: 各信号的平均Accuracy (%)')
    print('=' * 70)
    header = f'{"Signal":<10} {"Valence":>10} {"Arousal":>10} {"Dominance":>10}'
    print(header)
    print('-' * 70)
    for ft in feature_types:
        row = f'{ft.upper():<10}'
        for dim in dimensions:
            row += f' {results[ft][dim]["acc"]*100:>9.1f}%'
        print(row)

    print('\n' + '=' * 70)
    print('表7: 各信号的平均Balanced Accuracy (%)')
    print('=' * 70)
    print(header)
    print('-' * 70)
    for ft in feature_types:
        row = f'{ft.upper():<10}'
        for dim in dimensions:
            row += f' {results[ft][dim]["bacc"]*100:>9.1f}%'
        print(row)

    print('\n' + '=' * 70)
    print('表7: 各信号的平均F1-Score (%)')
    print('=' * 70)
    print(header)
    print('-' * 70)
    for ft in feature_types:
        row = f'{ft.upper():<10}'
        for dim in dimensions:
            row += f' {results[ft][dim]["f1"]*100:>9.1f}%'
        print(row)

    # 保存到文件
    summary_path = os.path.join(RESULTS_DIR, 'resultsTable.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('论文复现结果汇总\n')
        f.write('=' * 70 + '\n\n')
        for metric_name, metric_key in [('Accuracy', 'acc'), ('Balanced Accuracy', 'bacc'), ('F1-Score', 'f1')]:
            f.write(f'{metric_name} (%)\n')
            f.write(f'{"Signal":<10} {"Valence":>10} {"Arousal":>10} {"Dominance":>10}\n')
            f.write('-' * 50 + '\n')
            for ft in feature_types:
                row = f'{ft.upper():<10}'
                for dim in dimensions:
                    row += f' {results[ft][dim][metric_key]*100:>9.1f}%'
                f.write(row + '\n')
            f.write('\n')

    print(f'\n结果已保存到: {summary_path}')


def main():
    parser = argparse.ArgumentParser(description='论文复现流水线')
    parser.add_argument('--step', choices=['process', 'features', 'classify', 'summary'],
                        help='只运行指定步骤, 默认运行全部')
    args = parser.parse_args()

    steps = {
        'process': step_process,
        'features': step_features,
        'classify': step_classify,
        'summary': step_summary,
    }

    if args.step:
        steps[args.step]()
    else:
        step_process()
        step_features()
        step_classify()
        step_summary()


if __name__ == '__main__':
    main()
