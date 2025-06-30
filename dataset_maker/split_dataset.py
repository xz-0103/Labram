import os
import random
import shutil

def split_dataset(root_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    # 创建子文件夹
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    test_dir = os.path.join(root_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 获取所有文件（不包括文件夹）
    all_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    # 打乱文件顺序
    random.shuffle(all_files)

    # 计算每部分数量
    total = len(all_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    # 分配文件
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]

    # 移动文件
    for f in train_files:
        shutil.move(os.path.join(root_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.move(os.path.join(root_dir, f), os.path.join(val_dir, f))
    for f in test_files:
        shutil.move(os.path.join(root_dir, f), os.path.join(test_dir, f))

    print(f"✅ 数据集划分完成：共 {total} 个文件")
    print(f"   训练集: {len(train_files)}")
    print(f"   验证集: {len(val_files)}")
    print(f"   测试集: {len(test_files)}")

if __name__ == '__main__':
    # 替换成你的总文件夹路径
    folder_path = "../dataset_pkl2"
    split_dataset(folder_path)
