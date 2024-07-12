import torch
import pandas as pd
import os

import sys
sys.path.append('../')
import config

if config.REGION_NAME == 'CH':
    from config_ch import data_config,model_config
elif config.REGION_NAME == 'TX':
    from config_tx import data_config,model_config

def get_device():
    CUDA_NUM="0"
    CUDA_STR=f'cuda:{CUDA_NUM.split(",")[0]}'
    device = torch.device(CUDA_STR if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    return device
    

def get_model(model_name):
    device=get_device()
    # graph= torch.from_numpy(pd.read_excel(data_config.PEARSON_PATH).iloc[:,1:].values).type(torch.float32).to(device)
    graph= pd.read_excel(data_config.PEARSON_PATH).iloc[:,1:].to_numpy()
    if model_name=='APS_LSTM':
        from modelbase.APS_LSTM import APS_LSTM
        model_config.APS_LSTM_Args['adj_mx']=graph
        model=APS_LSTM(model_config.APS_LSTM_Args)
    else:
        raise BaseException(f"Your Choose Model: \"{model_name}\", But No Such Model!")
    model.to(device)
    return model,device