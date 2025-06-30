from pathlib import Path
from shock.utils import h5Dataset
from shock.utils import preprocessing_cnt

savePath = Path('path/to/your/save/path')
rawDataPath = Path('path/to/your/raw/data/path')
group = rawDataPath.glob('*.cnt') # 获取所有.cnt格式的EEG数据文件

# preprocessing parameters
l_freq = 0.1  # 低频截止，去除低频噪声。EEG 记录过程中，可能会因为电极接触问题、汗液、呼吸等因素导致 低频噪声，这些噪声通常在 0.1 Hz 以下。
h_freq = 75.0 #高频截止，去除高频噪声 。EEG 主要信号成分 在 0.1 Hz ~ 70 Hz 之间，而 肌电噪声（EMG） 主要集中在 70Hz 以上。
rsfreq = 200  # 重采样频率(Hz)

# channel number * rsfreq
# 设置HDF5数据集的块大小（62个通道×重采样频率）
chunks = (62, rsfreq)

dataset = h5Dataset(savePath, 'dataset')
for cntFile in group:
    # 打印当前处理文件名
    print(f'processing {cntFile.name}')
    # 读取并预处理 EEG 数据（滤波 & 重采样）
    eegData, chOrder = preprocessing_cnt(cntFile, l_freq, h_freq, rsfreq)
    # 将通道名称统一转换为大写
    chOrder = [s.upper() for s in chOrder]
    # 删除最后 10 秒的数据（去掉 10*200=2000 个采样点）
    eegData = eegData[:, :-10*rsfreq]
    # 在HDF5文件中创建与当前文件同名的组
    grp = dataset.addGroup(grpName=cntFile.stem)
    # 在该分组中添加 EEG 数据集
    dset = dataset.addDataset(grp, 'eeg', eegData, chunks)

    # dataset attributes
    # 添加数据集的属性信息（存储预处理参数）
    dataset.addAttributes(dset, 'lFreq', l_freq)
    dataset.addAttributes(dset, 'hFreq', h_freq)
    dataset.addAttributes(dset, 'rsFreq', rsfreq)
    dataset.addAttributes(dset, 'chOrder', chOrder)

dataset.save()
