import os
import random
import copy
import json

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import prepareData
from utils.metric import metric_mae,metric_rmse,metric_mape,masked_mape
import config
from utils.executer import get_model

if config.REGION_NAME == 'CH':
    from config_ch import data_config,model_config
elif config.REGION_NAME == 'TX':
    from config_tx import data_config,model_config


def set_random_seed(seed,deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)

set_random_seed(model_config.SEED,True)

MODEL_NAME='APS_LSTM'
model,device=get_model(MODEL_NAME)

CUDA_NUM="0"
CUDA_STR=f'cuda:{CUDA_NUM.split(",")[0]}'
device = torch.device(CUDA_STR if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

optimizer = torch.optim.Adam(params = model.parameters(), lr = model_config.lr)
mse_loss = torch.nn.MSELoss().to(device)

raw_data=pd.read_excel(data_config.DATA_PATH).to_numpy()
tr_data,va_data,te_data,data_tran=prepareData.split_data(raw_data,data_config.train_pre,data_config.valid_pre,\
                        data_config.seq_len,data_config.predict_len)

tr_bd=prepareData.baseDataset(tr_data)
va_bd=prepareData.baseDataset(va_data)
te_bd=prepareData.baseDataset(te_data)
tr_dl=DataLoader(tr_bd, batch_size=data_config.batch_size, shuffle=True)
va_dl=DataLoader(va_bd, batch_size=data_config.batch_size, shuffle=False)
te_dl=DataLoader(te_bd, batch_size=data_config.batch_size, shuffle=False)


def train_epoch(model, data_loader, optimizer, loss_fn):
    model.train()
    loss_record=[]
    for idx,(inputs,targets) in enumerate(data_loader):
        inputs=inputs.to(device)
        targets=targets.to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=loss_fn(outputs,targets)
        loss.backward()
        optimizer.step()
        loss_record.append(loss.item())
    return np.mean(loss_record)


def valid_epoch(model,data_loader,loss_fn):
    model.eval()
    loss_record=[]
    for idx,(inputs,targets) in enumerate(data_loader):
        inputs=inputs.to(device)
        targets=targets.to(device)
        with torch.no_grad():
            outputs=model(inputs)
            loss=loss_fn(outputs,targets)
            loss_record.append(loss.item())
    return np.mean(loss_record)


@torch.no_grad()
def test_epoch(model,data_loader,time_len:int,eval_metrics:dict,data_tran=None):
    model.eval()
    loss_records={i:{name:[] for name,_ in eval_metrics.items()} for i in range(time_len)}
     
    for idx,(inputs,targets) in enumerate(data_loader):
        inputs=inputs.to(device)
        targets=targets.to(device)
        outputs=model(inputs)
        if data_tran is not None:
            targets=data_tran.de_transform(targets)
            outputs=data_tran.de_transform(outputs)
        for T in range(time_len):
            for name,fn in eval_metrics.items():
                loss=fn(outputs[:,T],targets[:,T])
                loss_records[T][name].append(loss.item())

    for T in range(time_len):
        for name in eval_metrics.keys():
            loss_records[T][name]=np.mean(loss_records[T][name])
    
    print('\n')
    for T in range(time_len):
        str_builder=[]
        str_builder.extend((f'{name}:{val:.4f}' for name,val in loss_records[T].items()))
        str_builder=', '.join(str_builder)
        print(f"[Test Result T+{T+1}] {str_builder}")
    str_builder=[]
    loss_records['Avg']={}
    for name in loss_records[0].keys():
        loss_records['Avg'][name]=np.mean([loss_records[T][name] for T in range(time_len)])
        str_builder.append(f"{name}:{loss_records['Avg'][name]:.4f}")
    str_builder=', '.join(str_builder)
    print(f"[Test Result Avg] {str_builder}")

    # {0:{RMSE:xx,...},...,Avg:{RMSE:xx,...}}
    out_dir='./metric_record'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_path=f'{out_dir}/{MODEL_NAME}.json'
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(loss_records, f, ensure_ascii=False, indent=4)
    

def train(model, optimizer, loss_fn, early_stop=np.inf,  dir_path=None):
    best_loss = np.inf
    stop_count = 0
    for i in range(model_config.epoches):
        tr_loss=train_epoch(model,tr_dl,optimizer,loss_fn)
        va_loss=valid_epoch(model,va_dl,loss_fn)
        print(f'Epoch [{i+1}/{model_config.epoches}]: Train Loss: {tr_loss:.4f}, Valid Loss: {va_loss:.4f}')
        if va_loss<best_loss:
            best_loss = va_loss
            stop_count = 0
            best_model = copy.deepcopy(model)
            if dir_path:
                torch.save(model.state_dict(),rf'{dir_path}/{model.__class__.__name__}.pth')
        else:
            stop_count+=1
        if stop_count >= early_stop:
            print('\nThe Model is not improving, so we halt the training session.')
            return
    test_epoch(best_model,te_dl,config.PREDICT_LEN,{
        "RMSE": metric_rmse,
        "MAE": metric_mae,
        "MAPE": metric_mape,
        "masked_mape": masked_mape
    },data_tran)


if __name__ == '__main__':
    dir_path=f'model/{config.REGION_NAME}'
    train(model,optimizer,mse_loss,dir_path=dir_path)
    