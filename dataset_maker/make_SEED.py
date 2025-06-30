import os
import scipy.io
import numpy as np
import pickle
from scipy.signal import butter, filtfilt, iirnotch

input_dir = "../../dataset/SEED"        # 替换为你的 .mat 文件路径
output_dir = "../dataset_pkl2"
fixed_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

mapped_labels = [0 if x==-1 else (1 if x==0 else 2) for x in fixed_labels]

def bandpass_filter(data, lowcut=0.1, highcut=75.0, fs=200.0, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def notch_filter(data, notch_freq=50.0, fs=200.0, quality=30):
    b, a = iirnotch(notch_freq / (fs / 2), quality)
    return filtfilt(b, a, data, axis=-1)

def process_one_file(mat_file):
    file_path = os.path.join(input_dir, mat_file)
    filename = os.path.splitext(mat_file)[0]

    try:
        mat = scipy.io.loadmat(file_path)
        eeg_keys = [k for k in mat.keys() if 'eeg' in k.lower()]
        eeg_keys.sort()
        if len(eeg_keys) != 15:
            print(f"[!] {mat_file} 的 EEG 段数量为 {len(eeg_keys)}，不是 15 个，跳过")
            return

        os.makedirs(output_dir, exist_ok=True)

        for i, key in enumerate(eeg_keys):
            data = mat[key]
            if data.shape[0] > data.shape[1]:
                data = data.T  # [channel, time]

            # 滤波
            data = bandpass_filter(data, fs=200.0)
            data = notch_filter(data, fs=200.0)

            label = mapped_labels[i]

            # 每 2000 时间点切一段
            num_segments = data.shape[1] // 2000
            for j in range(num_segments):
                seg_data = data[:, j * 2000: (j + 1) * 2000]
                save_path = os.path.join(output_dir, f"{filename}_{i + 1}_{j + 1}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump({"X": seg_data.astype(np.float32), "y": label}, f)
                print(f"✅ Saved: {save_path}")
    except Exception as e:
        print(f"[✗] 处理 {mat_file} 失败: {e}")


if __name__ == "__main__":
    mat_files = [f for f in os.listdir(input_dir) if f.endswith(".mat")]
    for f in mat_files:
        print("🚀 正在处理", f)
        process_one_file(f)
