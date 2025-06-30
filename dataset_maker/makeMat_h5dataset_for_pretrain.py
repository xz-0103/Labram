from pathlib import Path
from shock.utils import h5Dataset
import scipy.io as sio
import numpy as np
import mne

savePath = Path('/home/test/members/Xu/LaBraM/dataset')
rawDataPath = Path('/home/test/members/Xu/data/SEED')
group = rawDataPath.glob('*.mat')  # 遍历SEED路径下的所有.mat文件

# 预处理参数
l_freq = 0.1
h_freq = 75.0
rsfreq = 200
chunks = (62, rsfreq)

# SEED数据标准的62通道名
standard_ch_names = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
    'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
    'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1',
    'OZ', 'O2', 'CB2'
]

def preprocessing_eeg(eegData, sfreq_ori, l_freq=0.1, h_freq=75.0, sfreq=200):
    # 用标准通道名（考虑可能某些通道缺失，做个安全截断）
    ch_names = standard_ch_names[:eegData.shape[0]]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq_ori, ch_types='eeg')
    raw = mne.io.RawArray(eegData, info)

    drop_ch = list(set(['M1', 'M2', 'VEO', 'HEO', 'ECG']) & set(ch_names))
    raw.drop_channels(drop_ch)

    raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
    raw = raw.notch_filter(50.0)

    if sfreq_ori != sfreq:
        raw = raw.resample(sfreq, n_jobs=5)

    eegData = raw.get_data()
    return eegData, raw.ch_names

dataset = h5Dataset(savePath, 'dataset')

for matFile in group:
    print(f'processing {matFile.name}')
    mat = sio.loadmat(matFile)

    for key in mat.keys():
        if 'eeg' in key.lower():
            print(f'processing {key}')
            eegData = mat[key]
            sfreq_ori = 200  # SEED原始采样率

            eegData, chOrder = preprocessing_eeg(eegData, sfreq_ori)

            chOrder = [s.upper() for s in chOrder]
            eegData = eegData[:, 1:-10 * rsfreq]  # 去掉最后10秒数据

            grpName = f'{matFile.stem}_{key}'
            grp = dataset.addGroup(grpName=grpName)
            dset = dataset.addDataset(grp, 'eeg', eegData, chunks)

            dataset.addAttributes(dset, 'lFreq', l_freq)
            dataset.addAttributes(dset, 'hFreq', h_freq)
            dataset.addAttributes(dset, 'rsFreq', rsfreq)
            dataset.addAttributes(dset, 'chOrder', chOrder)

dataset.save()
