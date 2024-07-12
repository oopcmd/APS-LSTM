import torch
import numpy as np
from torch.utils.data import Dataset


class DataTransform(object):
    def __init__(self) -> None:
        pass

    def transform(self,data:np.ndarray,tr_idx):
        tr_data=data[tr_idx[0][0]:tr_idx[-1][1]]
        self.tr_max=np.max(tr_data,axis=0,keepdims=True)
        self.tr_min=np.min(tr_data,axis=0,keepdims=True)
        def norm(x):
            x = 1. * (x - self.tr_min) / (self.tr_max - self.tr_min)
            x = 2. * x - 1.
            return x
        return norm(data)

    def de_transform(self,data:torch.Tensor):
        if not hasattr(self,'tr_max'):
            raise BaseException('Please use transform before using de_transform')
        @torch.no_grad()
        def denorm(x:torch.Tensor):
            x=(x+1)/2
            x=x*(self.tr_max[0,-1]-self.tr_min[0,-1])+self.tr_min[0,-1]
            return x
        return denorm(data)
            

def split_data(data,train_ratio,valid_ratio,history_seq_len,future_seq_len):
    length=data.shape[0]
    num_samples = length - (history_seq_len + future_seq_len)  + 1
    train_num_short = int(num_samples * train_ratio)
    valid_num_short = int(num_samples * valid_ratio)
    test_num_short  = num_samples - train_num_short - valid_num_short
    print("train_num_short:{0}".format(train_num_short))
    print("valid_num_short:{0}".format(valid_num_short))
    print("test_num_short:{0}".format(test_num_short))
    index_list = []
    for i in range(history_seq_len, num_samples + history_seq_len):
        index_list.append(
            (i - history_seq_len, i, i + future_seq_len)
        )
    train_idx = index_list[:train_num_short]
    vaild_idx = index_list[train_num_short:train_num_short + valid_num_short]
    test_idx = index_list[train_num_short + valid_num_short:train_num_short + valid_num_short + test_num_short]

    data_tran=DataTransform()
    data=data_tran.transform(data,train_idx)

    def split_from_idx(index):
        x_data,y_data=[],[]
        for idx1,idx2,idx3 in index:
            x_data.append(data[idx1:idx2])
            y_data.append(data[idx2:idx3,-1])
        return np.stack(x_data,axis=0),np.stack(y_data,axis=0)
    tr_x,tr_y=split_from_idx(train_idx)
    va_x,va_y=split_from_idx(vaild_idx)
    te_x,te_y=split_from_idx(test_idx)

    return (tr_x,tr_y),(va_x,va_y),(te_x,te_y),data_tran


class baseDataset(Dataset):
    def __init__(self, data_t):
        super(baseDataset, self).__init__()
        self.x,self.y=data_t
        self.length=self.x.shape[0]
        self.x=torch.from_numpy(self.x).float()
        self.y=torch.from_numpy(self.y).float()
        
    def __getitem__(self, index: int):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.length


    



