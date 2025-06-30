import pickle

# 读取 .pkl 文件
with open('dataset_pkl/train/1_20131027_1_1.pkl', 'rb') as f:
    data = pickle.load(f)

print(type(data))  # 查看数据类型
print(data)        # 输出内容
