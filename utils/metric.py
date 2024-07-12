import numpy as np
import torch

@torch.no_grad()
def metric_mse(reals, preds):
    loss = (preds - reals)**2
    return torch.mean(loss)

@torch.no_grad()
def metric_rmse(reals, preds):
    return torch.sqrt(metric_mse(reals, preds))

@torch.no_grad()
def metric_mae(reals, preds):
    loss = torch.abs(preds - reals)
    return torch.mean(loss)

@torch.no_grad()
def metric_mape(reals,preds):
    loss=torch.abs((preds-reals)/reals)
    return torch.mean(loss)


@torch.no_grad()
def masked_mape(reals, preds, null_val=1.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(reals)
    else:
        mask = ~torch.less_equal(torch.abs(reals),null_val)
    mask = mask.to(torch.float32)
    mask /= torch.mean(mask)
    mape = torch.abs(torch.divide(torch.subtract(preds, reals).to(torch.float32),reals))
    mape = torch.nan_to_num(mask * mape)
    return torch.mean(mape)
    
