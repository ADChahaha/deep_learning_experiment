import torch
import pandas as pd
import numpy as np

class CarDataset(torch.utils.data.Dataset):

    def __init__(self, path: str) -> None:
        super().__init__()
        origin_data = self.preprocess(path)
        self.label = origin_data["MPG"] 
        self.data = origin_data.drop("MPG", axis=1)
        self.label = torch.tensor(self.label.to_numpy(dtype=np.float32)).view(-1, 1)
        self.data = torch.tensor(self.data.to_numpy(dtype=np.float32))
        self.len = origin_data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return (self.data[index], self.label[index])

    def __len__(self) -> int:
        return self.len
        

    def preprocess(self, path):
        #使用pandas读取数据
        column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                        'Acceleration', 'Model Year', 'Origin']
        #遇到？换成nan，忽略\t之后的内容，已空格作为分隔符。
        raw_data = pd.read_csv(path, names=column_names,
                            na_values = "?", comment='\t',
                            sep=" ", skipinitialspace=True)

        data = raw_data.copy()
        #对于数据集中的空值，我们要在建模前进行处理。此处空值的数据较少，我们直接进行删除。
        #清洗空数据
        data = data.dropna()
        data.tail()
        #Pandas库提供了简单的数据集统计信息，我们可直接调用函数describe()进行查看。
        #查看训练数据集的结构
        origin = data.pop('Origin')
        data_labels = data.pop('MPG')
        train_stats = data.describe()
        train_stats = train_stats.transpose()
        #归一化数据
        def norm(x):
            return (x - train_stats['mean']) / train_stats['std']

        normed_data = norm(data)
        # 将MPG放回归一化后的数据中
        normed_data['MPG'] = data_labels 
        # 离散特征处理
        # 特征Origin代表着车辆的归属区域信息，此处总共分为三种，欧洲，美国，日本，我们需要对此特征进行one-hot编码。
        # 对origin属性进行one-hot编码
        normed_data['USA'] = (origin == 1)*1.0
        normed_data['Europe'] = (origin == 2)*1.0
        normed_data['Japan'] = (origin == 3)*1.0
        return normed_data 
